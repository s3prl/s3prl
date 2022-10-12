"""
The setting of Superb ASR

Authors
  * Heng-Jui Chang 2022
  * Leo 2022
"""

import logging
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import List

import pandas as pd
from omegaconf import MISSING
from torch.utils.data import Dataset

from s3prl.dataio.corpus.librispeech import LibriSpeech
from s3prl.dataio.dataset import EncodeText, LoadAudio, get_info
from s3prl.dataio.encoder.tokenizer import load_tokenizer
from s3prl.dataio.encoder.vocabulary import generate_vocab
from s3prl.dataio.sampler import FixedBatchSizeBatchSampler, SortedBucketingSampler
from s3prl.nn.rnn import RNNEncoder
from s3prl.nn.specaug import ModelWithSpecaug
from s3prl.util.download import urls_to_filepaths

from .run import ASR

logger = logging.getLogger(__name__)

__all__ = [
    "prepare_librispeech",
    "prepare_common_tokenizer",
    "SuperbASR",
]


def prepare_librispeech(
    target_dir,
    cache_dir,
    dataset_root,
    train_sets: List[str],
    valid_sets: List[str],
    test_sets: List[str],
    n_jobs: int = 6,
    get_path_only: bool = False,
):
    """
    Prepare LibriSpeech for ASR following :obj:`SuperbASR.prepare_data` format.
    See :obj:`LibriSpeech` for the arguments usage
    """
    target_dir = Path(target_dir)

    train_path = target_dir / f"{'+'.join(train_sets)}.csv"
    valid_path = target_dir / f"{'+'.join(valid_sets)}.csv"
    test_paths = [target_dir / f"{test_set}.csv" for test_set in test_sets]

    if get_path_only:
        return train_path, valid_path, test_paths

    corpus = LibriSpeech(dataset_root, n_jobs, train_sets, valid_sets, test_sets)
    train_data, valid_data, test_data = corpus.data_split

    def dict_to_csv(data_dict, csv_path):
        keys = sorted(list(data_dict.keys()))
        fields = sorted(data_dict[keys[0]].keys())
        data = dict()
        for field in fields:
            data[field] = []
            for key in keys:
                data[field].append(data_dict[key][field])
        data["id"] = keys
        df = pd.DataFrame(data)
        df.to_csv(csv_path, index=False)

    dict_to_csv(train_data, train_path)
    dict_to_csv(valid_data, valid_path)
    dict_to_csv(test_data, test_paths[0])

    return train_path, valid_path, test_paths


def prepare_common_tokenizer(
    target_dir,
    cache_dir,
    tokenizer_data_path,
    get_path_only=False,
    tokenizer_name: str = None,
    vocab_file: str = None,
    vocab_type: str = "character",
    vocab_args: dict = None,
    slots_file: str = None,
):
    """
    Build the tokenizer following :obj:`SuperbASR.build_tokenizer` format

    Args:
        tokenizer_name (str): Save the tokenizer filepath with this filename
        vocab_file (str): When the tokenizer was already prepared, and just want
            to load and return the tokenizer here. Path or URL
        vocab_type (str): character / phoneme / word / subword
        vocab_args (dict):
            when :code:`vocab_type` is character / phoneme / word, supports arguments in
                :obj:`s3prl.dataio.encoder.vocabulary.generate_basic_vocab`

            whe :code:`vocab_type` is subword, supports arguments in
                :obj:`s3prl.dataio.encoder.vocabulary.generate_subword_vocab`
        slots_file (str): If presented, the pre-defined slots will be used to encode the
            special tokens. Path or URL

    Return:
        str

        tokenizer path
    """
    if tokenizer_name is None:
        tokenizer_name = f"{Path(tokenizer_data_path).stem}-{vocab_type}.tokenizer"
    tokenizer_path = Path(target_dir) / f"{tokenizer_name}.pkl"

    if get_path_only:
        return tokenizer_path

    if vocab_file is not None:
        vocab_file = str(vocab_file)
        if vocab_file.startswith("http"):
            vocab_file = urls_to_filepaths(vocab_file)

    if slots_file is not None:
        slots_file = str(slots_file)
        if slots_file.startswith("http"):
            slots_file = urls_to_filepaths(slots_file)

    if vocab_file is not None:
        tokenizer = load_tokenizer(
            vocab_type,
            vocab_file=vocab_file,
            slots_file=slots_file,
        )
    else:
        vocab_args = vocab_args or {}
        assert isinstance(vocab_args, dict)

        if vocab_type == "subword" and not "output_file" in vocab_args:
            vocab_args["output_file"] = Path(target_dir) / "tokenizer.spm"

        vocab_result = generate_vocab(
            vocab_type, text_file=str(tokenizer_data_path), **vocab_args
        )
        vocab_list = vocab_result if isinstance(vocab_result, list) else None
        vocab_file = vocab_result if isinstance(vocab_result, str) else None
        tokenizer = load_tokenizer(
            vocab_type,
            vocab_file=vocab_file,
            vocab_list=vocab_list,
            slots_file=slots_file,
        )

    with open(tokenizer_path, "wb") as f:
        pickle.dump(tokenizer, f)

    return tokenizer_path


class SuperbASR(ASR):
    def default_config(self) -> dict:
        return dict(
            start=0,
            stop=None,
            target_dir=MISSING,
            cache_dir=None,
            remove_all_cache=False,
            prepare_data=dict(
                dataset_root=MISSING,
                train_sets=["train-clean-100"],
                valid_sets=["dev-clean"],
                test_sets=["test-clean"],
            ),
            prepare_tokenizer_data=dict(),
            build_tokenizer=dict(
                vocab_type="character",
            ),
            build_dataset=dict(),
            build_batch_sampler=dict(
                train=dict(
                    batch_size=32,
                    max_length=2000,  # due to this tiny max_length, the effective batch_size is always 16
                    shuffle=True,
                ),
                valid=dict(
                    batch_size=1,
                ),
                test=dict(
                    batch_size=1,
                ),
            ),
            build_upstream=dict(
                name=MISSING,
            ),
            build_featurizer=dict(
                layer_selections=None,
                normalize=False,
            ),
            build_downstream=dict(
                model_conf=dict(
                    module="LSTM",
                    proj_size=1024,
                    hidden_size=[1024, 1024],
                    dropout=[0.2, 0.2],
                    layer_norm=[False, False],
                    proj=[False, False],
                    sample_rate=[1, 1],
                    sample_style="concat",
                    bidirectional=True,
                ),
                specaug_conf=dict(
                    freq_mask_width_range=(0, 50),
                    num_freq_mask=4,
                    time_mask_width_range=(0, 40),
                    num_time_mask=2,
                ),
            ),
            build_model=dict(
                upstream_trainable=False,
            ),
            build_task=dict(
                log_metrics=["cer", "wer"],
            ),
            build_optimizer=dict(
                name="Adam",
                conf=dict(
                    lr=1.0e-4,
                ),
            ),
            build_scheduler=dict(
                name="ExponentialLR",
                gamma=0.9,
            ),
            save_model=dict(
                extra_conf=dict(
                    build_downstream_conf="${build_downstream}"
                ),  # This is redundant for ASR. Just to show how to clone other fields
            ),
            save_task=dict(),
            train=dict(
                total_steps=200000,
                log_step=100,
                eval_step=2000,
                save_step=500,
                gradient_clipping=1.0,
                gradient_accumulate=1,
                valid_metric="wer",
                valid_higher_better=False,
                auto_resume=True,
                resume_ckpt_dir=None,
            ),
        )

    def prepare_data(
        self,
        prepare_data: dict,
        target_dir: str,
        cache_dir: str,
        get_path_only: bool = False,
    ):
        """
        Prepare the task-specific data metadata (path, labels...).
        By default call :obj:`prepare_librispeech` with :code:`**prepare_data`

        Args:
            prepare_data (dict): same in :obj:`default_config`, support arguments in :obj:`prepare_librispeech`
            target_dir (str): Parse your corpus and save the csv file into this directory
            cache_dir (str): If the parsing or preprocessing takes too long time, you can save
                the temporary files into this directory. This directory is expected to be shared
                across different training sessions (different hypers and :code:`target_dir`)
            get_path_only (str): Directly return the filepaths no matter they exist or not.

        Returns:
            tuple

            1. train_path (str)
            2. valid_path (str)
            3. test_paths (List[str])

            Each path (str) should be a csv file containing the following columns:

            ====================  ====================
            column                description
            ====================  ====================
            id                    (str) - the unique id for this data point
            wav_path              (str) - the absolute path of the waveform file
            transcription         (str) - a text string
            ====================  ====================
        """
        return prepare_librispeech(
            **self._get_current_arguments(flatten_dict="prepare_data")
        )

    def prepare_tokenizer_data(
        self,
        prepare_tokenizer_data: dict,
        target_dir: str,
        cache_dir: str,
        train_csv: str,
        valid_csv: str,
        test_csvs: List[str],
        get_path_only: bool = False,
    ):
        """
        Prepare the text file used for training tokenizer.
        By default only use the transcription in the :code:`train_csv` returned from :obj:`prepare_data`
        The default :code:`prepare_tokenizer_data` build the character-based tokenizer

        Args:
            prepare_tokenizer_data (dict): same in :obj:`default_config`, no supported argument for now
            target_dir (str): Save the text file into this directory
            cache_dir (str): If the parsing or preprocessing takes too long time, you can save
                the temporary files into this directory. This directory is expected to be shared
                across different training sessions (different hypers and :code:`target_dir`)
            train_csv (str): The train data given by :obj:`prepare_data`
            get_path_only (str): Directly return the filepaths no matter they exist or not.

        Returns:
            str

            The text file path, the text file should be in the format

            .. code-block:: none

                This is the first line
                This is the second line
                These are all text used for training tokenizer

        """
        tokenizer_data_name = f"{Path(train_csv).stem}.tokenizer_data"
        tokenizer_data_path = Path(target_dir) / f"{tokenizer_data_name}.txt"

        if get_path_only:
            return tokenizer_data_path

        all_text = pd.read_csv(train_csv)["transcription"]

        with tokenizer_data_path.open("w") as f:
            f.writelines([f"{line}\n" for line in all_text])

        return tokenizer_data_path

    def build_tokenizer(
        self,
        build_tokenizer: dict,
        target_dir: str,
        cache_dir: str,
        tokenizer_data_path: str,
        get_path_only: bool = False,
    ):
        """
        Build the tokenizer from the data prepared by :obj:`prepare_tokenizer_data`
        By default call :obj:`prepare_common_tokenizer` with :code:`**build_tokenizer`

        Args:
            build_tokenizer (dict): same in :obj:`default_config`, arguments for :obj:`prepare_common_tokenizer`
            target_dir (str): Current experinment directory
            cache_dir (str): If the parsing or preprocessing takes too long time, you can save
                the temporary files into this directory. This directory is expected to be shared
                across different training sessions (different hypers and :code:`target_dir`)
            tokenizer_data_path (str): The text file from :obj:`prepare_tokenizer_data`
            get_path_only (str): Directly return the filepaths no matter they exist or not.

        Returns:
            str

            filepath of the pickled :obj:`s3prl.dataio.encoder.tokenizer.Tokenizer`
        """
        return prepare_common_tokenizer(
            **self._get_current_arguments(flatten_dict="build_tokenizer")
        )

    def build_dataset(
        self,
        build_dataset: dict,
        target_dir: str,
        cache_dir: str,
        mode: str,
        data_csv: str,
        tokenizer_path: str,
    ):
        """
        Build the dataset for train/valid/test.

        Args:
            build_dataset (dict): same in :obj:`default_config`, not used
            target_dir (str): Current experiment directory
            cache_dir (str): If the preprocessing takes too long time, you can save
                the temporary files into this directory. This directory is expected to be shared
                across different training sessions (different hypers and :code:`target_dir`)
            mode (str): train/valid/test
            data_csv (str): The metadata csv file for the specific :code:`mode`
            tokenizer_path (str): The pickled tokenizer path for encoding transcription

        Returns:
            torch Dataset

            For all train/valid/test mode, the dataset should return each item as a dictionary
            containing the following keys:

            ====================  ====================
            key                   description
            ====================  ====================
            x                     (torch.FloatTensor) - the waveform in (seq_len, 1)
            x_len                 (int) - the waveform length :code:`seq_len`
            class_ids             (torch.LongTensor) - the encoded class ids of a transcription (sentence)
            labels                (str) - the text transcription
            unique_name           (str) - the unique id for this datapoint
            ====================  ====================
        """
        csv = pd.read_csv(data_csv)

        audio_loader = LoadAudio(csv["wav_path"].tolist())

        with open(tokenizer_path, "rb") as f:
            tokenizer = pickle.load(f)

        text_encoder = EncodeText(csv["transcription"].tolist(), tokenizer)
        ids = csv["id"].tolist()

        class Speech2TextDataset:
            def __len__(self):
                return len(audio_loader)

            def __getitem__(self, index: int):
                audio = audio_loader[index]
                text = text_encoder[index]
                return {
                    "x": audio["wav"],
                    "x_len": audio["wav_len"],
                    "class_ids": text["class_ids"],
                    "labels": text["labels"],
                    "unique_name": ids[index],
                }

        dataset = Speech2TextDataset()
        return dataset

    def build_batch_sampler(
        self,
        build_batch_sampler: dict,
        target_dir: str,
        cache_dir: str,
        mode: str,
        data_csv: str,
        dataset: Dataset,
    ):
        """
        Return the batch sampler for torch DataLoader.

        Args:
            build_batch_sampler (dict): same in :obj:`default_config`

                ====================  ====================
                key                   description
                ====================  ====================
                train                 (dict) - arguments for :obj:`SortedBucketingSampler`
                valid                 (dict) - arguments for :obj:`FixedBatchSizeBatchSampler`
                test                  (dict) - arguments for :obj:`FixedBatchSizeBatchSampler`
                ====================  ====================

            target_dir (str): Current experiment directory
            cache_dir (str): If the preprocessing takes too long time, save
                the temporary files into this directory. This directory is expected to be shared
                across different training sessions (different hypers and :code:`target_dir`)
            mode (str): train/valid/test
            data_csv (str): the :code:`mode` specific csv from :obj:`prepare_data`
            dataset: the dataset from :obj:`build_dataset`

        Returns:
            batch sampler for torch DataLoader
        """

        @dataclass
        class Config:
            train: dict = None
            valid: dict = None
            test: dict = None

        conf = Config(**build_batch_sampler)

        if mode == "train":
            wav_lens = get_info(
                dataset, "x_len", cache_dir=Path(target_dir) / "train_stats"
            )
            sampler = SortedBucketingSampler(wav_lens, **(conf.train or {}))
        elif mode == "valid":
            sampler = FixedBatchSizeBatchSampler(dataset, **(conf.valid or {}))
        elif mode == "test":
            sampler = FixedBatchSizeBatchSampler(dataset, **(conf.test or {}))

        return sampler

    def build_downstream(
        self,
        build_downstream: dict,
        downstream_input_size: int,
        downstream_output_size: int,
        downstream_input_stride: int,
    ):
        """
        Return the task-specific downstream model.
        By default build the :obj:`RNNEncoder` model wrapped with :obj:`ModelWithSpecaug`

        Args:
            build_downstream (dict): same in :obj:`default_config`, has two keys:
                :code:`model_conf` is the arguments for :obj:`RNNEncoder`;
                :code:`specaug_conf` is the arguments for :obj:`ModelWithSpecaug`
            downstream_input_size (int): the required input size of the model
            downstream_output_size (int): the required output size of the model
            downstream_input_stride (int): the input feature's stride (from 16 KHz)

        Returns:
            :obj:`s3prl.nn.interface.AbsFrameModel`
        """

        @dataclass
        class Config:
            model_conf: dict = None
            specaug_conf: dict = None

        conf = Config(**build_downstream)
        model = RNNEncoder(
            downstream_input_size, downstream_output_size, **(conf.model_conf or {})
        )
        downstream = ModelWithSpecaug(model, **(conf.specaug_conf or {}))
        return downstream
