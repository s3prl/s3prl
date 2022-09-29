import logging
import os
import random
from dataclasses import dataclass
from typing import Callable, List

import torch
import torchaudio

from s3prl.dataio.encoder.category import CategoryEncoder
from s3prl.dataio.encoder.g2p import G2P
from s3prl.dataio.encoder.tokenizer import (
    Tokenizer,
    default_phoneme_tokenizer,
    load_tokenizer,
)
from s3prl.dataio.encoder.vocabulary import generate_vocab

from .base import AugmentedDynamicItemDataset, DataPipe

logger = logging.getLogger(__name__)


class SetOutputKeys(DataPipe):
    def __init__(self, output_keys: dict = None) -> None:
        super().__init__()
        self.output_keys = output_keys

    def forward(self, dataset: AugmentedDynamicItemDataset):
        dataset.update_output_keys(self.output_keys)
        return dataset


@dataclass
class LoadAudio(DataPipe):
    audio_sample_rate: int = 16000
    audio_channel_reduction: str = "first"
    sox_effects: list = None

    wav_path_name: str = "wav_path"
    wav_name: str = "wav"
    start_sec_name: str = "start_sec"
    end_sec_name: str = "end_sec"

    def load_audio(
        self,
        wav_path,
        start_sec: float = None,
        end_sec: float = None,
    ):
        crop_segment = start_sec is not None and end_sec is not None

        torchaudio.set_audio_backend("sox_io")
        wav, sr = torchaudio.load(
            wav_path,
            frame_offset=round(start_sec * self.audio_sample_rate)
            if crop_segment
            else 0,
            num_frames=round((end_sec - start_sec) * self.audio_sample_rate)
            if crop_segment
            else -1,
        )

        if self.sox_effects is not None:
            wav, sr = torchaudio.sox_effects.apply_effects_tensor(
                wav, sr, effects=self.sox_effects
            )

        if sr != self.audio_sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.audio_sample_rate)
            wav = resampler(wav)

        if self.audio_channel_reduction == "first":
            wav = wav[0]
        elif self.audio_channel_reduction == "mean":
            wav = wav.mean(dim=0)

        wav = wav.view(-1, 1)
        return wav

    def compute_length(self, wav):
        return len(wav)

    def forward(self, dataset: AugmentedDynamicItemDataset):
        item = dataset[0]
        if self.start_sec_name in item and self.end_sec_name in item:
            crop_segment = True
        else:
            crop_segment = False

        if not crop_segment:
            dataset.add_dynamic_item(
                self.load_audio, takes=self.wav_path_name, provides=self.wav_name
            )
        else:
            dataset.add_dynamic_item(
                self.load_audio,
                takes=[self.wav_path_name, self.start_sec_name, self.end_sec_name],
                provides=self.wav_name,
            )
        dataset.add_dynamic_item(
            self.compute_length,
            takes=self.wav_name,
            provides=f"{self.wav_name}_len",
        )
        return dataset


@dataclass
class EncodeCategory(DataPipe):
    train_category_encoder: bool = False
    label_name: str = "label"
    category_encoder_name: str = "category"
    encoded_target_name: str = "class_id"

    def prepare_category(self, labels):
        return CategoryEncoder(sorted(list(set(labels))))

    def encode_label(self, category, label):
        return category.encode(label)

    def forward(self, dataset: AugmentedDynamicItemDataset):
        if self.train_category_encoder:
            with dataset.output_keys_as([self.label_name]):
                labels = [item[self.label_name] for item in dataset]
            category = self.prepare_category(labels)
            dataset.add_tool(self.category_encoder_name, category)

        category = dataset.get_tool(self.category_encoder_name)
        dataset.add_tool("output_size", len(category))

        dataset.add_dynamic_item(
            self.encode_label,
            takes=[self.category_encoder_name, self.label_name],
            provides=self.encoded_target_name,
        )
        return dataset


@dataclass
class EncodeMultipleCategory(EncodeCategory):
    train_category_encoder: bool = False
    label_name: str = "labels"
    category_encoder_name: str = "categories"
    encoded_target_name: str = "class_ids"

    def encode_label(self, categories, labels):
        return torch.LongTensor(
            [category.encode(label) for category, label in zip(categories, labels)]
        )

    def forward(self, dataset: AugmentedDynamicItemDataset):
        if self.train_category_encoder:
            with dataset.output_keys_as([self.label_name]):
                labels = [item[self.label_name] for item in dataset]
            label_types = list(zip(*labels))
            categories = [
                self.prepare_category(label_type) for label_type in label_types
            ]
            dataset.add_tool(self.category_encoder_name, categories)
            dataset.add_tool("output_size", sum([len(c) for c in categories]))

        dataset.add_dynamic_item(
            self.encode_label,
            takes=[self.category_encoder_name, self.label_name],
            provides=self.encoded_target_name,
        )
        return dataset


@dataclass
class EncodeMultiLabel(DataPipe):
    label_name: str = "labels"
    category_encoder_name: str = "category"
    encoded_target_name: str = "binary_labels"

    @staticmethod
    def label_to_binary_vector(label: List, num_labels: int) -> torch.Tensor:
        # Lame special case for multilabel with no labels
        if len(label) == 0:
            # BCEWithLogitsLoss wants float not long targets
            binary_labels = torch.zeros((num_labels,), dtype=torch.float)
        else:
            binary_labels = torch.zeros((num_labels,)).scatter(
                0, torch.tensor(label), 1.0
            )

        # Validate the binary vector we just created
        assert set(torch.where(binary_labels == 1.0)[0].numpy()) == set(label)
        return binary_labels

    def encode_label(self, category, labels):
        labels = [category.encode(label) for label in labels]
        binary_labels = self.label_to_binary_vector(labels, len(category))
        return binary_labels

    def forward(self, dataset: AugmentedDynamicItemDataset):
        if not dataset.has_tool(self.category_encoder_name):
            with dataset.output_keys_as([self.label_name]):
                all_labels = []
                for item in dataset:
                    all_labels.extend(item[self.label_name])
                all_labels.sort()
                all_labels = set(all_labels)
                category = CategoryEncoder(all_labels)
                dataset.add_tool(self.category_encoder_name, category)

        category = dataset.get_tool(self.category_encoder_name)
        dataset.add_tool("output_size", len(category))

        dataset.add_dynamic_item(
            self.encode_label,
            takes=[self.category_encoder_name, self.label_name],
            provides=self.encoded_target_name,
        )
        return dataset


@dataclass
class GenerateTokenizer(DataPipe):
    generate: bool = True
    tokenizer_name: str = "tokenizer"
    text_name: str = "transcription"
    vocab_type: str = "character"
    text_file: str = None
    vocab_file: str = None
    slots_file: str = None
    vocab_args: dict = None

    def prepare_tokenizer(self, text_list: str = None) -> Tokenizer:
        """Generates tokenizer from text data.

        Args:
            text_list (str, optional): List of text. Defaults to None.

        Returns:
            Tokenizer: Generated tokenizer
        """

        vocab_args = self.vocab_args or {}
        assert isinstance(vocab_args, dict)

        if text_list is not None:
            vocab_result = generate_vocab(
                self.vocab_type, text_list=text_list, **vocab_args
            )
        else:
            vocab_result = generate_vocab(
                self.vocab_type, text_file=self.text_file, **vocab_args
            )
        vocab_list = vocab_result if isinstance(vocab_result, list) else None
        vocab_file = vocab_result if isinstance(vocab_result, str) else None

        tokenizer = load_tokenizer(
            self.vocab_type,
            vocab_file=vocab_file,
            vocab_list=vocab_list,
            slots_file=self.slots_file,
        )
        return tokenizer

    def forward(self, dataset: AugmentedDynamicItemDataset):
        try:
            tokenizer = dataset.get_tool(self.tokenizer_name)
            logger.info(
                f"Tokenizer (name = {self.tokenizer_name}) exists in dataset, skip generation."
            )
        except KeyError:
            if self.generate:
                if self.vocab_file is not None and os.path.exists(self.vocab_file):
                    tokenizer = load_tokenizer(
                        self.vocab_type,
                        vocab_file=self.vocab_file,
                        slots_file=self.slots_file,
                    )
                else:
                    text_list = None
                    if self.text_file is None:
                        with dataset.output_keys_as([self.text_name]):
                            text_list = [item[self.text_name] for item in dataset]

                    tokenizer = self.prepare_tokenizer(text_list)

                dataset.add_tool(self.tokenizer_name, tokenizer)
            else:
                logger.warning(
                    "No tokenizer is found or generated. No-op for this DataPipe"
                )

        return dataset


@dataclass
class EncodeText(DataPipe):
    text_name: str = "transcription"
    output_text_name: str = "tokenized_text"
    tokenizer_name: str = "tokenizer"

    def encode_text(self, tokenizer: Tokenizer, text: str) -> torch.LongTensor:
        return torch.LongTensor(tokenizer.encode(text))

    def forward(self, dataset: AugmentedDynamicItemDataset):
        try:
            tokenizer = dataset.get_tool(self.tokenizer_name)
        except KeyError:
            raise KeyError(f"Tokenizer (name = {self.tokenizer_name}) not found!")

        dataset.add_dynamic_item(
            self.encode_text,
            takes=[self.tokenizer_name, self.text_name],
            provides=self.output_text_name,
        )
        dataset.add_tool("output_size", tokenizer.vocab_size)

        return dataset


@dataclass
class Phonemize(DataPipe):
    text_name: str = "transcription"
    phonemized_text_name: str = "phonemized_text"
    output_text_name: str = "tokenized_text"
    g2p_name: str = "g2p"
    tokenizer_name: str = "tokenizer"

    def grapheme2phoneme(self, g2p: G2P, text: str) -> str:
        return g2p.encode(text)

    def encode_text(self, tokenizer: Tokenizer, text: str) -> torch.LongTensor:
        return torch.LongTensor(tokenizer.encode(text))

    def forward(self, dataset: AugmentedDynamicItemDataset):
        if not dataset.has_tool(self.g2p_name):
            logger.warning(
                f"Cannot find {self.g2p_name} in dataset, use default G2P instead."
            )
            dataset.add_tool(self.g2p_name, G2P())

        if not dataset.has_tool(self.tokenizer_name):
            logger.warning(
                f"Cannot find {self.tokenizer_name} in dataset, use default tokenizer instead."
            )
            dataset.add_tool(self.tokenizer_name, default_phoneme_tokenizer())

        dataset.add_dynamic_item(
            self.grapheme2phoneme,
            takes=[self.g2p_name, self.text_name],
            provides=self.phonemized_text_name,
        )

        dataset.add_dynamic_item(
            self.encode_text,
            takes=[self.tokenizer_name, self.phonemized_text_name],
            provides=self.output_text_name,
        )

        tokenizer = dataset.get_tool(self.tokenizer_name)
        dataset.add_tool("output_size", tokenizer.vocab_size)

        return dataset


@dataclass
class RandomCrop(DataPipe):
    """
    Completely randomized for every batch even with the same datapoint id.
    Only suitable for training.
    """

    sample_rate: int = 16000
    max_secs: float = None

    wav_name: str = "wav"
    crop_name: str = "wav_crop"

    def crop_wav(self, wav):
        if self.max_secs is not None and wav.size(0) > self.max_secs * self.sample_rate:
            start = random.randint(0, wav.size(0) - self.max_secs * self.sample_rate)
            end = start + self.max_secs * self.sample_rate
            wav = wav[round(start) : round(end)]
        return wav, wav.size(0)

    def forward(
        self, dataset: AugmentedDynamicItemDataset
    ) -> AugmentedDynamicItemDataset:
        dataset.add_dynamic_item(
            self.crop_wav,
            takes=[self.wav_name],
            provides=[self.crop_name, f"{self.crop_name}_len"],
        )
        return dataset
