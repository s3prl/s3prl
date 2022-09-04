import logging
import pickle
from collections import OrderedDict
from pathlib import Path
from typing import List

import pandas as pd
from omegaconf import MISSING
from torch.utils.data import Dataset

from s3prl.corpus.librispeech import LibriSpeech
from s3prl.dataset.speech2text_pipe import Speech2TextPipe
from s3prl.encoder.tokenizer import load_tokenizer
from s3prl.encoder.vocabulary import generate_vocab
from s3prl.nn.interface import AbsFrameModel
from s3prl.nn.rnn import RNNEncoder
from s3prl.nn.specaug import ModelWithSpecaug
from s3prl.sampler import FixedBatchSizeBatchSampler, SortedBucketingSampler

from .run import ASR

logger = logging.getLogger(__name__)


def prepare_common_tokenizer(
    _target_dir,
    _cache_dir,
    _tokenizer_data_path,
    _get_path_only=False,
    tokenizer_name: str = None,
    vocab_file: str = None,
    vocab_type: str = "character",
    vocab_args: dict = None,
    slots_file: str = None,
):
    if tokenizer_name is None:
        tokenizer_name = f"{Path(_tokenizer_data_path).stem}-{vocab_type}.tokenizer"
    tokenizer_path = Path(_target_dir) / f"{tokenizer_name}.pkl"

    if _get_path_only:
        return tokenizer_path

    if vocab_file is not None:
        tokenizer = load_tokenizer(
            vocab_type,
            vocab_file=vocab_file,
            slots_file=slots_file,
        )
    else:
        vocab_args = vocab_args or {}
        assert isinstance(vocab_args, dict)

        vocab_result = generate_vocab(
            vocab_type, text_file=str(_tokenizer_data_path), **vocab_args
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
    @classmethod
    def default_config(cls) -> dict:
        return dict(
            start=0,
            stop=None,
            target_dir=MISSING,
            cache_dir=str(Path.home() / ".cache" / "s3prl" / "data"),
            remove_all_cache=False,
            prepare_data=dict(
                dataset_root=MISSING,
                train_set="train-clean-100",
                valid_set="dev-clean",
                test_sets=["test-clean"],
            ),
            prepare_tokenizer_data=dict(),
            build_tokenizer=dict(
                tokenizer_name=None,
                vocab_type="character",
                vocab_args=None,
                slots_file=None,
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
                name="fbank",
            ),
            build_featurizer=dict(
                layer_selections=None,
                normalize=False,
            ),
            build_downstream=dict(
                model_cfg=dict(
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
                specaug_cfg=dict(
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
                gradient_accumulate_steps=1,
                valid_metric="wer",
                valid_higher_better=False,
                auto_resume=True,
                resume_ckpt_dir=None,
            ),
        )

    @classmethod
    def prepare_data(
        cls,
        _target_dir,
        _cache_dir,
        dataset_root,
        train_set: str,
        valid_set: str,
        test_sets: List[str],
        n_jobs: int = 6,
        _get_path_only=False,
    ):
        target_dir = Path(_target_dir)

        train_path = target_dir / f"{train_set}.csv"
        valid_path = target_dir / f"{valid_set}.csv"
        test_paths = [target_dir / f"{test_set}.csv" for test_set in test_sets]

        if _get_path_only:
            return train_path, valid_path, test_paths

        corpus = LibriSpeech(dataset_root, n_jobs, [train_set], [valid_set], test_sets)
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

    @classmethod
    def prepare_tokenizer_data(
        cls, _target_dir, _cache_dir, _train_csv, _get_path_only=False
    ):
        tokenizer_data_name = f"{Path(_train_csv).stem}.tokenizer_data"
        tokenizer_data_path = Path(_target_dir) / f"{tokenizer_data_name}.txt"

        if _get_path_only:
            return tokenizer_data_path

        all_text = pd.read_csv(_train_csv)["transcription"]

        with tokenizer_data_path.open("w") as f:
            f.writelines([f"{line}\n" for line in all_text])

        return tokenizer_data_path

    @classmethod
    def build_tokenizer(
        cls,
        _target_dir,
        _cache_dir,
        _tokenizer_data_path,
        _get_path_only=False,
        **config,
    ):
        return prepare_common_tokenizer(
            _target_dir,
            _cache_dir,
            _tokenizer_data_path,
            _get_path_only=_get_path_only,
            **config,
        )

    @classmethod
    def build_dataset(
        cls,
        _target_dir: str,
        _cache_dir: str,
        _mode: str,
        _data_csv: str,
        _tokenizer_path: str,
    ):
        data_points = OrderedDict()
        csv = pd.read_csv(_data_csv)
        for _, row in csv.iterrows():
            data_points[row["id"]] = {
                "wav_path": row["wav_path"],
                "transcription": row["transcription"],
            }

        with open(_tokenizer_path, "rb") as f:
            tokenizer = pickle.load(f)

        dataset = Speech2TextPipe(generate_tokenizer=False)(
            data_points,
            tools={"tokenizer": tokenizer},
        )
        return dataset

    @classmethod
    def build_batch_sampler(
        cls,
        _target_dir: str,
        _cache_dir: str,
        _mode: str,
        _data_csv: str,
        _dataset: Dataset,
        train: dict = None,
        valid: dict = None,
        test: dict = None,
    ):
        train = train or {}
        valid = valid or {}
        test = test or {}

        if _mode == "train":
            sampler = SortedBucketingSampler(_dataset, **train)
        elif _mode == "valid":
            sampler = FixedBatchSizeBatchSampler(_dataset, **valid)
        elif _mode == "test":
            sampler = FixedBatchSizeBatchSampler(_dataset, **test)

        return sampler

    @classmethod
    def build_downstream(
        cls,
        _downstream_input_size: int,
        _downstream_output_size: int,
        _downstream_downsample_rate: int,
        model_cfg: dict,
        specaug_cfg: dict,
    ) -> AbsFrameModel:
        model = RNNEncoder(_downstream_input_size, _downstream_output_size, **model_cfg)
        downstream = ModelWithSpecaug(model, **specaug_cfg)
        return downstream
