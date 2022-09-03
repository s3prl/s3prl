import pandas as pd
from pathlib import Path
from omegaconf import MISSING

from s3prl.util.download import _urls_to_filepaths
from s3prl.sampler import FixedBatchSizeBatchSampler
from s3prl.corpus.snips import SNIPS

from .superb_asr import SuperbASR

VOCAB_URL = "https://huggingface.co/datasets/s3prl/SNIPS/raw/main/character.txt"
SLOTS_URL = "https://huggingface.co/datasets/s3prl/SNIPS/raw/main/slots.txt"


class SuperbSF(SuperbASR):
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
                train_speakers=[
                    "Ivy",
                    "Joanna",
                    "Joey",
                    "Justin",
                    "Kendra",
                    "Kimberly",
                    "Matthew",
                    "Salli",
                ],
                valid_speakers=["Aditi", "Amy", "Geraint", "Nicole"],
                test_speakers=["Brian", "Emma", "Raveena", "Russell"],
            ),
            prepare_tokenizer_data=dict(),
            build_tokenizer=dict(
                tokenizer_name=None,
                vocab_type="character",
                vocab_file=_urls_to_filepaths(VOCAB_URL),
                slots_file=_urls_to_filepaths(SLOTS_URL),
            ),
            build_dataset=dict(),
            build_batch_sampler=dict(
                train=dict(
                    batch_size=32,
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
                log_metrics=[
                    "wer",
                    "cer",
                    "slot_type_f1",
                    "slot_value_cer",
                    "slot_value_wer",
                    "slot_edit_f1_full",
                    "slot_edit_f1_part",
                ],
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
            save_model=dict(),
            save_task=dict(),
            train=dict(
                total_steps=200000,
                log_step=100,
                eval_step=2000,
                save_step=500,
                gradient_clipping=1.0,
                gradient_accumulate_steps=1,
                valid_metric="slot_type_f1",
                valid_higher_better=True,
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
        train_speakers,
        valid_speakers,
        test_speakers,
        _get_path_only=False,
    ):
        target_dir = Path(_target_dir)

        train_path = target_dir / f"train.csv"
        valid_path = target_dir / f"valid.csv"
        test_paths = [target_dir / f"test.csv"]

        if _get_path_only:
            return train_path, valid_path, test_paths

        corpus = SNIPS(dataset_root, train_speakers, valid_speakers, test_speakers)
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
        train = train or {}
        valid = valid or {}
        test = test or {}

        if _mode == "train":
            return FixedBatchSizeBatchSampler(_dataset, **train)
        elif _mode == "valid":
            return FixedBatchSizeBatchSampler(_dataset, **valid)
        elif _mode == "test":
            return FixedBatchSizeBatchSampler(_dataset, **test)
