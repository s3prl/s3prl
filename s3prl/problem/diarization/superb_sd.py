from pathlib import Path
from omegaconf import MISSING

from s3prl.dataset.diarization import DiarizationDataset
from s3prl.sampler import FixedBatchSizeBatchSampler, GroupSameItemSampler
from s3prl.nn.interface import AbsFrameModel
from s3prl.nn.rnn import SuperbDiarizationModel

from .run import Diarization


class SuperbSD(Diarization):
    @classmethod
    def default_config(cls):
        return dict(
            start=0,
            stop=None,
            target_dir=MISSING,
            cache_dir=str(Path.home() / ".cache" / "s3prl" / "data"),
            remove_all_cache=False,
            prepare_data=dict(
                dataset_root=MISSING,
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
                    batch_size=32,
                    shuffle=True,
                ),
                valid=dict(
                    batch_size=1,
                ),
                test=dict(
                    meta_name="record_id",
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
                total_steps=200000,
                log_step=100,
                eval_step=2000,
                save_step=500,
                gradient_clipping=1.0,
                gradient_accumulate_steps=1,
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

    @classmethod
    def prepare_data(
        cls, target_dir, cache_dir, dataset_root: str, _get_path_only=False
    ):
        train_dir = Path(dataset_root) / "train"
        valid_dir = Path(dataset_root) / "dev"
        test_dir = Path(dataset_root) / "test"
        return train_dir, valid_dir, [test_dir]

    @classmethod
    def build_dataset(
        cls,
        _target_dir: str,
        _cache_dir: str,
        _mode: str,
        _data_dir: str,
        _num_speakers: int,
        _frame_shift: int,
        **config,
    ):
        dataset = DiarizationDataset(
            _mode,
            _data_dir,
            frame_shift=_frame_shift,
            num_speakers=_num_speakers,
            **config,
        )
        return dataset

    @classmethod
    def build_batch_sampler(
        cls,
        _target_dir,
        _cache_dir,
        _mode,
        _data_dir,
        _dataset,
        train: dict = None,
        valid: dict = None,
        test: dict = None,
    ):
        train = train or {}
        valid = valid or {}
        test = test or {}

        if _mode == "train":
            return FixedBatchSizeBatchSampler(_dataset, **train)
        elif _mode == "valid":
            return FixedBatchSizeBatchSampler(_dataset, **valid)
        elif _mode == "test":
            return GroupSameItemSampler(_dataset, **test)

    @classmethod
    def build_downstream(
        cls,
        _downstream_input_size: int,
        _downstream_output_size: int,
        _downstream_downsample_rate: int,
        **config,
    ) -> AbsFrameModel:
        return SuperbDiarizationModel(
            _downstream_input_size, _downstream_output_size, **config
        )
