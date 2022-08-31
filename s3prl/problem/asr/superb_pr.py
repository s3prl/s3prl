import pickle
from pathlib import Path
from collections import OrderedDict

import pandas as pd
from omegaconf import MISSING

from s3prl.dataset.speech2phoneme_pipe import Speech2PhonemePipe
from s3prl.encoder.tokenizer import default_phoneme_tokenizer
from s3prl.nn.linear import FrameLevelLinear
from s3prl.sampler import FixedBatchSizeBatchSampler, SortedSliceSampler

from .superb_asr import SuperbASR


class SuperbPR(SuperbASR):
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
            build_tokenizer=dict(),
            build_dataset=dict(),
            build_batch_sampler=dict(
                train=dict(
                    batch_size=16,
                    max_length=300000,
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
                hidden_size=256,
            ),
            build_model=dict(
                upstream_trainable=False,
            ),
            build_task=dict(
                log_metrics=["per"],
            ),
            build_optimizer=dict(
                name="Adam",
                conf=dict(
                    lr=1.0e-2,
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
                valid_metric="per",
                valid_higher_better=False,
                auto_resume=True,
                resume_ckpt_dir=None,
            ),
            evaluate=dict(),
        )

    @classmethod
    def build_tokenizer(
        cls,
        _target_dir,
        _cache_dir,
        _tokenizer_data_path,
        _get_path_only=False,
    ):
        tokenizer_path = Path(_target_dir) / "default_phone_tokenizer.pkl"
        with tokenizer_path.open("wb") as f:
            pickle.dump(default_phoneme_tokenizer(), f)
        return tokenizer_path

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

        dataset = Speech2PhonemePipe()(data_points, tokenizer=tokenizer)
        return dataset

    @classmethod
    def build_batch_sampler(
        cls,
        _target_dir: str,
        _cache_dir: str,
        _mode: str,
        _data_csv: str,
        _dataset,
        train: dict = None,
        valid: dict = None,
        test: dict = None,
    ):
        if _mode == "train":
            sampler = SortedSliceSampler(_dataset, **train)
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
        hidden_size: int,
    ):
        return FrameLevelLinear(
            _downstream_input_size, _downstream_output_size, hidden_size=hidden_size
        )
