from fairseq.tasks.speech_to_text import SpeechToTextTask
from fairseq.data import FairseqDataset, data_utils, ResamplingDataset, ConcatDataset
from .Fairseq_SpeechToTextDataset import SpeechToTextDataset, SpeechToTextDatasetCreator, S2TDataConfig
from typing import Dict, List, Optional, Tuple
import torchaudio
import torch
from argparse import Namespace
import os.path as op
import csv


# the following codes are modify from fairseq's implementation
# (https://github.com/pytorch/fairseq)
class S3prl_SpeechToTextTask(SpeechToTextTask):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def load_dataset(self, split, max_feature_len = -1, epoch=1, combine=False, **kwargs):
        is_train_split = split.startswith("train")
        pre_tokenizer = self.build_tokenizer(self.args)
        bpe_tokenizer = self.build_bpe(self.args)
        self.datasets[split] = S3prl_SpeechToTextDatasetCreator.from_tsv(
            self.args.data,
            self.data_cfg,
            split,
            self.tgt_dict,
            pre_tokenizer,
            bpe_tokenizer,
            is_train_split=is_train_split,
            epoch=epoch,
            seed=self.args.seed,
            upstream_rate = self.upstream_rate,
            max_feature_len = max_feature_len,
        )
    
    def build_model(self, args, input_dim):
        args.input_feat_per_channel = input_dim
        args.input_channels = self.data_cfg.input_channels
        return super(SpeechToTextTask, self).build_model(args)

class S3prl_SpeechToTextDatasetCreator(SpeechToTextDatasetCreator):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    # optional
    KEY_SAMPLE_RATE = 'sr'

    # default
    DEFAULT_SAMPLE_RATE = 16000

    @classmethod
    def _from_list(
        cls,
        split_name: str,
        is_train_split,
        samples: List[List[Dict]],
        data_cfg: S2TDataConfig,
        tgt_dict,
        pre_tokenizer,
        bpe_tokenizer,
        upstream_rate,
        max_feature_len,
    ) -> SpeechToTextDataset:
        audio_paths, n_frames, src_texts, tgt_texts, ids = [], [], [], [], []
        speakers, src_langs, tgt_langs = [], [], []
        srs = []
        for s in samples:
            ids.extend([ss[cls.KEY_ID] for ss in s])
            audio_paths.extend(
                [op.join(data_cfg.audio_root, ss[cls.KEY_AUDIO]) for ss in s]
            )
            n_frames.extend([int(ss[cls.KEY_N_FRAMES]) for ss in s])
            tgt_texts.extend([ss[cls.KEY_TGT_TEXT] for ss in s])
            src_texts.extend(
                [ss.get(cls.KEY_SRC_TEXT, cls.DEFAULT_SRC_TEXT) for ss in s]
            )
            speakers.extend([ss.get(cls.KEY_SPEAKER, cls.DEFAULT_SPEAKER) for ss in s])
            src_langs.extend([ss.get(cls.KEY_SRC_LANG, cls.DEFAULT_LANG) for ss in s])
            tgt_langs.extend([ss.get(cls.KEY_TGT_LANG, cls.DEFAULT_LANG) for ss in s])
            
            # sample rate
            srs.extend([int(ss.get(cls.KEY_SAMPLE_RATE, cls.DEFAULT_SAMPLE_RATE)) for ss in s])
        
        return S3prl_SpeechToTextDataset(
            split_name,
            is_train_split,
            data_cfg,
            audio_paths,
            n_frames,
            src_texts = src_texts,
            tgt_texts = tgt_texts,
            speakers = speakers,
            src_langs = src_langs,
            tgt_langs = tgt_langs,
            srs = srs,
            ids = ids,
            tgt_dict = tgt_dict,
            pre_tokenizer = pre_tokenizer,
            bpe_tokenizer = bpe_tokenizer,
            upstream_rate = upstream_rate,
            max_feature_len = max_feature_len,
        )

    @classmethod
    def from_tsv(
        cls,
        root: str,
        data_cfg: S2TDataConfig,
        splits: str,
        tgt_dict,
        pre_tokenizer,
        bpe_tokenizer,
        is_train_split: bool,
        epoch: int,
        seed: int,
        upstream_rate: int,
        max_feature_len: int,
    ) -> SpeechToTextDataset:

        _splits = splits.split(",")
        assert len(_splits) == 1, "do not support multiple files training"

        split = _splits[0]
        tsv_path = op.join(root, f"{split}.tsv")
        if not op.isfile(tsv_path):
            raise FileNotFoundError(f"Dataset not found: {tsv_path}")
        
        with open(tsv_path) as f:
            reader = csv.DictReader(
                f,
                delimiter="\t",
                quotechar=None,
                doublequote=False,
                lineterminator="\n",
                quoting=csv.QUOTE_NONE,
            )
            samples = [dict(e) for e in reader]

        assert len(samples) > 0

        return cls._from_list(
            split,
            is_train_split,
            [samples],
            data_cfg,
            tgt_dict,
            pre_tokenizer,
            bpe_tokenizer,
            upstream_rate,
            max_feature_len,
        )


class S3prl_SpeechToTextDataset(SpeechToTextDataset):

    TARGET_RATE = 16000

    def __init__(self, *args, srs = Optional[List[int]], upstream_rate = 160, max_feature_len=-1, **kwargs):
        
        super().__init__(*args, **kwargs)

        self.srs = srs
        self.max_feature_len = max_feature_len
        self.max_wav_len = max_feature_len * upstream_rate
        self.resamplers = {}
        for sr in set(srs):
            self.resamplers[sr] = torchaudio.transforms.Resample(
                orig_freq = sr,
                new_freq = self.TARGET_RATE,
            )

        for i in range(len(self.n_frames)):
            new_n_frames = self.n_frames[i] * self.TARGET_RATE / self.srs[i] / upstream_rate
            if self.max_feature_len > 0 and new_n_frames > max_feature_len:
                new_n_frames = max_feature_len
            self.n_frames[i] = int(new_n_frames)

    def __getitem__(
        self, index: int
    ) -> Tuple[str, int, torch.Tensor, Optional[torch.Tensor]]:

        source, sr = torchaudio.load(self.audio_paths[index])

        assert self.srs[index] == sr
        source = self.resamplers[sr](source)
        source = torch.mean(source, dim=0)
        source = source.view(-1)

        if self.feature_transforms is not None:
            assert not self.data_cfg.use_audio_input
            source = self.feature_transforms(source)
        source = source.float()

        # truncate the wav
        if self.max_feature_len > 0:
            if source.size(0) > self.max_wav_len:
                print(f'wav too long({source.size(0)}), truncate to {self.max_wav_len} (id={index})')
                source = source[:self.max_wav_len]

        target = None
        if self.tgt_texts is not None:
            tokenized = self.tokenize_text(self.tgt_texts[index])
            target = self.tgt_dict.encode_line(
                tokenized, add_if_not_exist=False, append_eos=True
            ).long()
            if self.data_cfg.prepend_tgt_lang_tag:
                lang_tag = self.LANG_TAG_TEMPLATE.format(self.tgt_langs[index])
                lang_tag_idx = self.tgt_dict.index(lang_tag)
                target = torch.cat((torch.LongTensor([lang_tag_idx]), target), 0)
        return self.ids[index], index, source, target

    def collater(self, samples: List[Tuple[str, int, torch.Tensor, torch.Tensor]]):
        ids = [sample[0] for sample in samples]
        samples = [sample[1:] for sample in samples]
        output_dict = super().collater(samples)
        output_dict['utt_id'] = ids

        wavs = []

        for i in range(output_dict['nsentences']):
            wav = output_dict['net_input']['src_tokens'][i]
            length = output_dict['net_input']['src_lengths'][i].item()
            wavs.append(wav[:length].numpy())

        return wavs, output_dict
