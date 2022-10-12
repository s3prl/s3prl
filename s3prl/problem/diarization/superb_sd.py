"""
The setting fo Superb SD

Authors:
  * Jiatong Shi 2021
  * Leo 2022
"""

from dataclasses import dataclass
from pathlib import Path

from omegaconf import MISSING

from s3prl.dataio.dataset import DiarizationDataset, get_info
from s3prl.dataio.sampler import FixedBatchSizeBatchSampler, GroupSameItemSampler
from s3prl.nn.rnn import SuperbDiarizationModel

from .run import Diarization
from .util import kaldi_dir_to_csv

__all__ = [
    "SuperbSD",
]


class SuperbSD(Diarization):
    def default_config(self):
        return dict(
            start=0,
            stop=None,
            target_dir=MISSING,
            cache_dir=None,
            remove_all_cache=False,
            prepare_data=dict(
                data_dir=MISSING,
            ),
            build_dataset=dict(
                chunk_size=2000,
                subsampling=1,
                rate=16000,
                use_last_samples=True,
                label_delay=0,
            ),
            build_batch_sampler=dict(
                train=dict(
                    batch_size=8,
                    shuffle=True,
                ),
                valid=dict(
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
                hidden_size=512,
                rnn_layers=1,
            ),
            build_model=dict(
                upstream_trainable=False,
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
                total_steps=30000,
                log_step=500,
                eval_step=500,
                save_step=500,
                gradient_clipping=1.0,
                gradient_accumulate=4,
                valid_metric="der",
                valid_higher_better=False,
                auto_resume=True,
                resume_ckpt_dir=None,
            ),
            scoring=dict(
                thresholds=[0.3, 0.4, 0.5, 0.6, 0.7],
                median_filters=[1, 11],
            ),
        )

    def prepare_data(
        self, prepare_data: dict, target_dir: str, cache_dir: str, get_path_only=False
    ):
        """
        Prepare the task-specific data metadata (path, labels...).

        Args:
            prepare_data (dict): same in :obj:`default_config`

                ====================  ====================
                key                   description
                ====================  ====================
                data_dir              (str) - the standard Kaldi data directory
                ====================  ====================

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
            record_id             (str) - the id for the recording
            duration              (float) - the total seconds of the recording
            wav_path              (str) - the absolute path of the recording
            utt_id                (str) - the id for the segmented utterance, should be \
                                    globally unique across all recordings instead of just \
                                    unique in a recording
            speaker               (str) - the speaker label for the segmented utterance
            start_sec             (float) - segment start second in the recording
            end_sec               (float) - segment end second in the recording
            ====================  ====================

            Instead of one waveform file per row, the above file format is one segment per row,
            and a waveform file can have multiple overlapped segments uttered by different speakers.
        """

        @dataclass
        class Config:
            data_dir: str

        conf = Config(**prepare_data)

        target_dir: Path = Path(target_dir)
        train_csv = target_dir / "train.csv"
        valid_csv = target_dir / "valid.csv"
        test_csv = target_dir / "test.csv"

        if get_path_only:
            return train_csv, valid_csv, [test_csv]

        kaldi_dir_to_csv(Path(conf.data_dir) / "train", train_csv)
        kaldi_dir_to_csv(Path(conf.data_dir) / "dev", valid_csv)
        kaldi_dir_to_csv(Path(conf.data_dir) / "test", test_csv)
        return train_csv, valid_csv, [test_csv]

    def build_dataset(
        self,
        build_dataset: dict,
        target_dir: str,
        cache_dir: str,
        mode: str,
        data_csv: str,
        data_dir: str,
        num_speakers: int,
        frame_shift: int,
    ):
        """
        Build the dataset for train/valid/test.

        Args:
            build_dataset (dict): same in :obj:`default_config`, supports arguments for :obj:`DiarizationDataset`
            target_dir (str): Current experiment directory
            cache_dir (str): If the preprocessing takes too long time, you can save
                the temporary files into this directory. This directory is expected to be shared
                across different training sessions (different hypers and :code:`target_dir`)
            mode (str): train/valid/test
            data_csv (str): The metadata csv file for the specific :code:`mode`
            data_dir (str): The converted kaldi data directory from :code:`data_csv`
            num_speakers (int): The number of speaker per utterance
            frame_shift (int): The frame shift of the upstream model (downsample rate from 16 KHz)

        Returns:
            torch Dataset

            For all train/valid/test mode, the dataset should return each item as a dictionary
            containing the following keys:

            ====================  ====================
            key                   description
            ====================  ====================
            x                     (torch.FloatTensor) - the waveform in (seq_len, 1)
            x_len                 (int) - the waveform length :code:`seq_len`
            label                 (torch.LongTensor) - the binary label for each upstream frame, \
                                    shape: :code:`(upstream_len, 2)`
            label_len             (int) - the upstream feature's seq length :code:`upstream_len`
            record_id             (str) - the unique id for the recording
            chunk_id              (int) - since recording can be chunked into several segments \
                                    for efficient training, this field indicate the segment's \
                                    original position (order, 0-index) in the recording. This \
                                    field is only useful during the testing stage
            ====================  ====================
        """

        dataset = DiarizationDataset(
            mode,
            data_dir,
            frame_shift=frame_shift,
            num_speakers=num_speakers,
            **build_dataset,
        )
        return dataset

    def build_batch_sampler(
        self,
        build_batch_sampler: dict,
        target_dir: str,
        cache_dir: str,
        mode: str,
        data_csv: str,
        data_dir: str,
        dataset,
    ):
        """
        Return the batch sampler for torch DataLoader.

        Args:
            build_batch_sampler (dict): same in :obj:`default_config`

                ====================  ====================
                key                   description
                ====================  ====================
                train                 (dict) - arguments for :obj:`FixedBatchSizeBatchSampler`
                valid                 (dict) - arguments for :obj:`FixedBatchSizeBatchSampler`
                test                  (dict) - arguments for :obj:`GroupSameItemSampler`, should always \
                                        use this batch sampler for the testing stage
                ====================  ====================

            target_dir (str): Current experiment directory
            cache_dir (str): If the preprocessing takes too long time, save
                the temporary files into this directory. This directory is expected to be shared
                across different training sessions (different hypers and :code:`target_dir`)
            mode (str): train/valid/test
            data_csv (str): The metadata csv file for the specific :code:`mode`
            data_dir (str): The converted kaldi data directory from :code:`data_csv`
            dataset: the dataset from :obj:`build_dataset`

        Returns:
            batch sampler for torch DataLoader
        """

        @dataclass
        class Config:
            train: dict = None
            valid: dict = None

        conf = Config(**build_batch_sampler)

        if mode == "train":
            return FixedBatchSizeBatchSampler(dataset, **(conf.train or {}))
        elif mode == "valid":
            return FixedBatchSizeBatchSampler(dataset, **(conf.valid or {}))
        elif mode == "test":
            record_ids = get_info(dataset, "record_id")
            return GroupSameItemSampler(record_ids)
        else:
            raise ValueError(f"Unsupported mode: {mode}")

    def build_downstream(
        self,
        build_downstream: dict,
        downstream_input_size: int,
        downstream_output_size: int,
        downstream_input_stride: int,
    ):
        """
        Return the task-specific downstream model.
        By default build the :obj:`SuperbDiarizationModel` model

        Args:
            build_downstream (dict): same in :obj:`default_config`, support arguments of :obj:`SuperbDiarizationModel`
            downstream_input_size (int): the required input size of the model
            downstream_output_size (int): the required output size of the model
            downstream_input_stride (int): the input feature's stride (from 16 KHz)

        Returns:
            :obj:`s3prl.nn.interface.AbsFrameModel`
        """
        return SuperbDiarizationModel(
            downstream_input_size, downstream_output_size, **build_downstream
        )
