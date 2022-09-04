from collections import OrderedDict
import pickle
import pandas as pd
from pathlib import Path
from omegaconf import MISSING

from s3prl.corpus.voxceleb1sv import VoxCeleb1SV
from s3prl.dataset.speaker_verification_pipe import SpeakerVerificationPipe
from s3prl.sampler import FixedBatchSizeBatchSampler
from s3prl.nn.speaker_model import SuperbXvector
from s3prl.util.download import _download

from .run import ASV

EFFECTS = [
    ["channels", "1"],
    ["rate", "16000"],
    ["gain", "-3.0"],
    ["silence", "1", "0.1", "0.1%", "-1", "0.1", "0.1%"],
]


def prepare_voxceleb1_for_sv(
    _target_dir: str,
    _cache_dir: str,
    _get_path_only: str,
    dataset_root: str,
    force_download: bool = True,
):
    train_path = _target_dir / "train.csv"
    test_trial_path = _target_dir / "test_trial.csv"

    if _get_path_only:
        return train_path, [test_trial_path]

    corpus = VoxCeleb1SV(dataset_root, _cache_dir, force_download)
    train_data, valid_data, test_data, test_trials = corpus.all_data
    all_data = {**train_data, **valid_data}

    ignored_utts_path = Path(_cache_dir) / "voxceleb1_too_short_utts"
    _download(
        ignored_utts_path,
        "https://huggingface.co/datasets/s3prl/voxceleb1_too_short_utts/raw/main/utt",
        True,
    )
    with open(ignored_utts_path) as file:
        ignored_utts = [line.strip() for line in file.readlines()]

    for utt in ignored_utts:
        assert utt in all_data
        del all_data[utt]

    ids = sorted(all_data.keys())
    wav_paths = [all_data[idx]["wav_path"] for idx in ids]
    labels = [all_data[idx]["label"] for idx in ids]
    pd.DataFrame({"id": ids, "wav_path": wav_paths, "spk": labels}).to_csv(
        train_path, index=False
    )

    labels, id1s, id2s = zip(*test_trials)
    wav_path1 = [test_data[idx]["wav_path"] for idx in id1s]
    wav_path2 = [test_data[idx]["wav_path"] for idx in id2s]
    pd.DataFrame(
        {
            "id1": id1s,
            "id2": id2s,
            "wav_path1": wav_path1,
            "wav_path2": wav_path2,
            "label": labels,
        }
    ).to_csv(test_trial_path, index=False)

    return train_path, [test_trial_path]


class SuperbASV(ASV):
    @classmethod
    def default_config(cls):
        return dict(
            target_dir=MISSING,
            cache_dir=str(Path.home() / ".cache" / "s3prl" / "asv"),
            test_ckpt_steps=[
                20000,
                40000,
                60000,
                80000,
                100000,
                120000,
                140000,
                160000,
                180000,
                200000,
            ],
            prepare_data=dict(
                dataset_root=MISSING,
            ),
            build_dataset=dict(
                train=dict(
                    random_crop_secs=8.0,
                ),
            ),
            build_batch_sampler=dict(
                train=dict(
                    batch_size=10,
                    shuffle=True,
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
            build_model=dict(
                upstream_trainable=False,
            ),
            build_task=dict(
                loss_type="amsoftmax",
                loss_cfg=dict(
                    margin=0.4,
                    scale=30,
                ),
            ),
            build_optimizer=dict(
                name="AdamW",
                conf=dict(
                    lr=1.0e-4,
                ),
            ),
            build_scheduler=dict(
                name="ExponentialLR",
                gamma=0.9,
            ),
            train=dict(
                total_steps=200000,
                log_step=500,
                eval_step=1e20,
                save_step=20000,
                gradient_clipping=1.0e3,
                gradient_accumulate_steps=5,
                valid_metric=None,
                valid_higher_better=None,
                auto_resume=True,
                resume_ckpt_dir=None,
                keep_num_ckpts=10,
            ),
        )

    @classmethod
    def prepare_data(
        cls, _target_dir: str, _cache_dir: str, _get_path_only: bool, **prepare_data
    ):
        return prepare_voxceleb1_for_sv(
            _target_dir, _cache_dir, _get_path_only, **prepare_data
        )

    @classmethod
    def build_encoder(
        cls,
        _target_dir: str,
        _cache_dir: str,
        _train_csv: str,
        _test_csvs: str,
        _get_path_only: bool,
    ):
        encoder_path = Path(_target_dir) / "spk2int.pkl"
        if _get_path_only:
            return encoder_path

        csv = pd.read_csv(_train_csv)
        all_spk = sorted(set(csv["spk"]))

        spk2int = {}
        for idx, spk in enumerate(all_spk):
            spk2int[spk] = idx

        with open(encoder_path, "wb") as f:
            pickle.dump(spk2int, f)

        return encoder_path

    @classmethod
    def build_dataset(
        cls,
        _target_dir,
        _cache_dir,
        _mode,
        _data_csv,
        _encoder_path,
        train: dict = None,
        test: dict = None,
    ):
        train = train or {}
        test = test or {}

        if _mode == "train":
            csv = pd.read_csv(_data_csv)
            data = OrderedDict()
            for rowid, row in csv.iterrows():
                data[row["id"]] = dict(
                    wav_path=row["wav_path"],
                    label=row["spk"],
                )
            return SpeakerVerificationPipe(sox_effects=EFFECTS, **train)(data)
        elif _mode == "test":
            csv = pd.read_csv(_data_csv)
            ids = pd.concat([csv["id1"], csv["id2"]], ignore_index=True).tolist()
            wav_paths = pd.concat([csv["wav_path1"], csv["wav_path2"]], ignore_index=True).tolist()
            data_list = sorted(set([(idx, path) for idx, path in zip(ids, wav_paths)]))
            data = OrderedDict()
            for idx, path in data_list:
                data[idx] = dict(
                    wav_path=path,
                    label=None,
                )
            return SpeakerVerificationPipe(sox_effects=EFFECTS, **test)(data)

    @classmethod
    def build_batch_sampler(
        cls,
        _target_dir,
        _cache_dir,
        _mode,
        _data_csv,
        _dataset,
        train,
        test,
    ):
        train = train or {}
        test = test or {}

        if _mode == "train":
            return FixedBatchSizeBatchSampler(_dataset, **train)
        elif _mode == "test":
            return FixedBatchSizeBatchSampler(_dataset, **test)

    @classmethod
    def build_downstream(
        cls,
        _downstream_input_size: int,
        _downstream_output_size: int,
        _downstream_downsample_rate: int,
        **build_downstream,
    ):
        return SuperbXvector(
            _downstream_input_size, _downstream_output_size, **build_downstream
        )
