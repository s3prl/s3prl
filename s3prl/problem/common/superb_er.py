"""
The setting of Superb ER

Authors
  * Tzu-Hsien Huang 2021
  * Leo 2021
  * Leo 2022
"""

import json
import logging
from pathlib import Path
from typing import List

import pandas as pd
import torch
from omegaconf import MISSING
from torch.utils.data import random_split

from s3prl.dataio.corpus.iemocap import IEMOCAP
from s3prl.util.download import download

from .superb_sid import SuperbSID

logger = logging.getLogger(__name__)

__all__ = [
    "iemocap_for_superb",
    "SuperbER",
]


def iemocap_for_superb(
    target_dir: str,
    cache_dir: str,
    iemocap: str,
    test_fold: int,
    valid_ratio: float = 0.2,
    get_path_only: bool = False,
):
    """
    Prepare IEMOCAP for emotion classfication with SUPERB protocol,
    following :obj:`SuperbER.prepare_data` format.

    .. note::

        In SUPERB protocol, you need to do 5-fold cross validation.

        Also, only use 4 emotion classes: :code:`happy`, :code:`angry`,
        :code:`neutral`, and :code:`sad` with balanced data points and
        the :code:`excited` class is merged into :code:`happy` class.

    Args:
        iemocap (str): The root path of the IEMOCAP
        test_fold (int): Which fold to use as the test fold, select from 0 to 4
        valid_ratio (float): given the remaining 4 folds, how many data to use as the validation set
        **others: refer to :obj:`SuperbER.prepare_data`
    """
    target_dir = Path(target_dir)

    train_path = target_dir / f"train.csv"
    valid_path = target_dir / f"valid.csv"
    test_paths = [target_dir / f"test.csv"]

    if get_path_only:
        return train_path, valid_path, test_paths

    corpus = IEMOCAP(iemocap)
    all_datapoints = corpus.all_data

    def format_fields(data: dict):
        result = dict()
        for data_id in data.keys():
            datapoint = data[data_id]
            result[data_id] = dict(
                wav_path=datapoint["wav_path"],
                label=datapoint["emotion"],
            )
        return result

    def filter_data(data_ids: List[str]):
        result = dict()
        for data_id in data_ids:
            data_point = all_datapoints[data_id]
            if data_point["emotion"] in ["neu", "hap", "ang", "sad", "exc"]:
                if data_point["emotion"] == "exc":
                    data_point["emotion"] = "hap"
                result[data_id] = data_point
        return result

    test_session_id = test_fold + 1
    train_meta_data_json = (
        Path(cache_dir) / f"test_session{test_session_id}_train_metadata.json"
    )
    test_meta_data_json = (
        Path(cache_dir) / f"test_session{test_session_id}_test_metadata.json"
    )
    download(
        train_meta_data_json,
        f"https://huggingface.co/datasets/s3prl/iemocap_split/raw/4097f2b496c41eed016d4e5eb0ada4cccd46d1f3/Session{test_session_id}/train_meta_data.json",
        refresh=False,
    )
    download(
        test_meta_data_json,
        f"https://huggingface.co/datasets/s3prl/iemocap_split/raw/4097f2b496c41eed016d4e5eb0ada4cccd46d1f3/Session{test_session_id}/test_meta_data.json",
        refresh=False,
    )

    with open(train_meta_data_json) as f:
        metadata = json.load(f)["meta_data"]

    dev_ids = [Path(item["path"]).stem for item in metadata]

    with open(test_meta_data_json) as f:
        metadata = json.load(f)["meta_data"]

    test_ids = [Path(item["path"]).stem for item in metadata]

    train_len = int((1 - valid_ratio) * len(dev_ids))
    train_valid_lens = [train_len, len(dev_ids) - train_len]

    torch.manual_seed(0)
    train_ids, valid_ids = random_split(dev_ids, train_valid_lens)

    train_data = format_fields(filter_data(train_ids))
    valid_data = format_fields(filter_data(valid_ids))
    test_data = format_fields(filter_data(test_ids))

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


class SuperbER(SuperbSID):
    def default_config(self) -> dict:
        return dict(
            start=0,
            stop=None,
            target_dir=MISSING,
            cache_dir=None,
            remove_all_cache=False,
            prepare_data=dict(
                iemocap=MISSING,
                test_fold=MISSING,
            ),
            build_encoder=dict(),
            build_dataset=dict(),
            build_batch_sampler=dict(
                train=dict(
                    batch_size=4,
                    shuffle=True,
                ),
                valid=dict(
                    batch_size=4,
                ),
                test=dict(
                    batch_size=4,
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
            build_task=dict(),
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
                total_steps=30000,
                log_step=500,
                eval_step=1000,
                save_step=1000,
                gradient_clipping=1.0,
                gradient_accumulate=8,
                valid_metric="accuracy",
                valid_higher_better=True,
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
        """
        Prepare the task-specific data metadata (path, labels...).
        By default call :obj:`iemocap_for_superb` with :code:`**prepare_data`

        Args:
            prepare_data (dict): same in :obj:`default_config`,
                support arguments in :obj:`iemocap_for_superb`
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
            label                 (str) - a string label of the waveform
            start_sec             (float) - optional, load the waveform from :code:`start_sec` seconds. If not presented or is :code:`math.nan`, load from the beginning.
            end_sec               (float) - optional, load the waveform from :code:`end_sec` seconds. If not presented or is :code:`math.nan`, load to the end.
            ====================  ====================
        """
        return iemocap_for_superb(
            **self._get_current_arguments(flatten_dict="prepare_data")
        )
