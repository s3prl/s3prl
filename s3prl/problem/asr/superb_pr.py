"""
The setting of Superb PR

Authors
  * Heng-Jui Chang 2022
  * Leo 2022
"""

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from omegaconf import MISSING

from s3prl.dataio.dataset import get_info
from s3prl.dataio.encoder.g2p import G2P
from s3prl.dataio.sampler import FixedBatchSizeBatchSampler, SortedSliceSampler
from s3prl.nn.linear import FrameLevelLinear

from .superb_asr import SuperbASR

__all__ = [
    "SuperbPR",
]


class SuperbPR(SuperbASR):
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
                vocab_type="phoneme",
            ),
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
                name=MISSING,
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
                total_steps=100000,
                log_step=100,
                eval_step=1000,
                save_step=100,
                gradient_clipping=1.0,
                gradient_accumulate=2,
                valid_metric="per",
                valid_higher_better=False,
                auto_resume=True,
                resume_ckpt_dir=None,
            ),
            evaluate=dict(),
        )

    def prepare_data(
        self,
        prepare_data: dict,
        target_dir: str,
        cache_dir: str,
        get_path_only: bool = False,
    ):
        train_csv, valid_csv, test_csvs = super().prepare_data(
            prepare_data, target_dir, cache_dir, get_path_only
        )
        if get_path_only:
            return train_csv, valid_csv, test_csvs

        g2p = G2P()

        def phonemize(csv_path):
            df = pd.read_csv(csv_path)
            text = df["transcription"].tolist()
            phonemized_text = [g2p.encode(t.strip()) for t in text]
            df["transcription"] = phonemized_text
            df.to_csv(csv_path, index=False)

        for csv_path in [train_csv, valid_csv, *test_csvs]:
            phonemize(csv_path)

        return train_csv, valid_csv, test_csvs

    def build_batch_sampler(
        self,
        build_batch_sampler: dict,
        target_dir: str,
        cache_dir: str,
        mode: str,
        data_csv: str,
        dataset,
    ):
        """
        Return the batch sampler for torch DataLoader.

        Args:
            build_batch_sampler (dict): same in :obj:`default_config`

                ====================  ====================
                key                   description
                ====================  ====================
                train                 (dict) - arguments for :obj:`SortedSliceSampler`
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
            sampler = SortedSliceSampler(wav_lens, **(conf.train or {}))
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
        By default build the :obj:`FrameLevelLinear`

        Args:
            build_downstream (dict): same in :obj:`default_config`,
                supports arguments in :obj:`FrameLevelLinear`
            downstream_input_size (int): the required input size of the model
            downstream_output_size (int): the required output size of the model
            downstream_input_stride (int): the input feature's stride (from 16 KHz)

        Returns:
            :obj:`s3prl.nn.interface.AbsFrameModel`
        """
        return FrameLevelLinear(
            downstream_input_size, downstream_output_size, **build_downstream
        )
