import tempfile

import pytest
import yaml
from dotenv import dotenv_values
from torch.utils.data import Subset
from tqdm import tqdm

from s3prl.downstream.emotion.expert import DownstreamExpert
from s3prl.problem import SuperbER


@pytest.mark.corpus
@pytest.mark.parametrize("fold_id", [0, 1, 2, 3, 4])
def test_er_dataset(fold_id):

    IEMOCAP = dotenv_values()["IEMOCAP"]
    with open("./s3prl/downstream/emotion/config.yaml") as file:
        config = yaml.load(file, Loader=yaml.FullLoader)["downstream_expert"]
        config["datarc"]["root"] = IEMOCAP
        config["datarc"]["meta_data"] = "./s3prl/downstream/emotion/meta_data"
        config["datarc"]["test_fold"] = f"fold{fold_id + 1}"

    with tempfile.TemporaryDirectory() as tempdir:
        expert = DownstreamExpert(320, config, tempdir)

        train_dataset_v3 = expert.get_dataloader("train").dataset
        valid_dataset_v3 = expert.get_dataloader("dev").dataset
        test_dataset_v3 = expert.get_dataloader("test").dataset

    with tempfile.TemporaryDirectory() as tempdir:
        default_config = SuperbER().default_config()
        train_csv, valid_csv, test_csvs = SuperbER().prepare_data(
            {"iemocap": IEMOCAP, "test_fold": fold_id}, tempdir, tempdir
        )
        encoder_path = SuperbER().build_encoder(
            default_config["build_encoder"],
            tempdir,
            tempdir,
            train_csv,
            valid_csv,
            test_csvs,
        )
        train_dataset_v4 = SuperbER().build_dataset(
            default_config["build_dataset"],
            tempdir,
            tempdir,
            "train",
            train_csv,
            encoder_path,
        )
        valid_dataset_v4 = SuperbER().build_dataset(
            default_config["build_dataset"],
            tempdir,
            tempdir,
            "valid",
            valid_csv,
            encoder_path,
        )
        test_dataset_v4 = SuperbER().build_dataset(
            default_config["build_dataset"],
            tempdir,
            tempdir,
            "test",
            test_csvs[0],
            encoder_path,
        )

    def compare_dataset(v3, v4):
        data_v3 = {}
        for wav, label, name in tqdm(v3, desc="v3"):
            if isinstance(v3, Subset):
                v3 = v3.dataset
            label_name = [k for k, v in v3.class_dict.items() if v == label][0]
            data_v3[name] = label_name

        data_v4 = {}
        for batch in tqdm(v4, desc="v4"):
            data_v4[batch["unique_name"]] = batch["label"]

        assert sorted(data_v3.keys()) == sorted(data_v4.keys())
        for key in data_v3:
            value_v3 = data_v3[key]
            value_v4 = data_v4[key]
            assert value_v3 == value_v4

    compare_dataset(train_dataset_v3, train_dataset_v4)
    compare_dataset(valid_dataset_v3, valid_dataset_v4)
    compare_dataset(test_dataset_v3, test_dataset_v4)
