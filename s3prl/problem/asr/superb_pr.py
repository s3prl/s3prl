"""
The setting of Superb PR

Authors
  * Heng-Jui Chang 2022
  * Shu-wen Yang 2022
"""

import pickle
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from omegaconf import MISSING

from s3prl.dataset.speech2phoneme_pipe import Speech2PhonemePipe
from s3prl.dataio.encoder.tokenizer import default_phoneme_tokenizer
from s3prl.nn.linear import FrameLevelLinear
from s3prl.dataio.sampler import FixedBatchSizeBatchSampler, SortedSliceSampler

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
                gradient_accumulate=1,
                valid_metric="per",
                valid_higher_better=False,
                auto_resume=True,
                resume_ckpt_dir=None,
            ),
            evaluate=dict(),
        )

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
        By default use the :obj:`default_phoneme_tokenizer`

        Args:
            build_tokenizer (dict): same in :obj:`default_config`, not used
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

        tokenizer_path = Path(target_dir) / "default_phone_tokenizer.pkl"
        with tokenizer_path.open("wb") as f:
            pickle.dump(default_phoneme_tokenizer(), f)
        return tokenizer_path

    def build_dataset(
        self,
        build_dataset: dict,
        target_dir: str,
        cache_dir: str,
        mode: str,
        data_csv: str,
        tokenizer_path: str,
    ):
        data_points = OrderedDict()
        csv = pd.read_csv(data_csv)
        for _, row in csv.iterrows():
            data_points[row["id"]] = {
                "wav_path": row["wav_path"],
                "transcription": row["transcription"],
            }

        with open(tokenizer_path, "rb") as f:
            tokenizer = pickle.load(f)

        dataset = Speech2PhonemePipe()(
            data_points,
            tools={"tokenizer": tokenizer},
        )
        return dataset

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
            sampler = SortedSliceSampler(dataset, **(conf.train or {}))
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
