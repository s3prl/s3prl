import pytest
import yaml
from tqdm import tqdm
from dotenv import dotenv_values

from s3prl.problem import SuperbER
from s3prl.downstream.emotion.expert import DownstreamExpert
from torch.utils.data import Subset

@pytest.mark.corpus
@pytest.mark.parametrize("fold_id", [0, 1, 2, 3, 4])
def test_er_dataset(fold_id):
    IEMOCAP = dotenv_values()["IEMOCAP"]
    with open("./s3prl/downstream/emotion/config.yaml") as file:
        config = yaml.load(file)["downstream_expert"]
        config["datarc"]["root"] = IEMOCAP
        config["datarc"]["meta_data"] = "./s3prl/downstream/emotion/meta_data"
        config["datarc"]["test_fold"] = f"fold{fold_id + 1}"
    expert = DownstreamExpert(320, config, "result/tmp")

    train_dataset_v3 = expert.get_dataloader("train").dataset
    valid_dataset_v3 = expert.get_dataloader("dev").dataset
    test_dataset_v3 = expert.get_dataloader("test").dataset

    train_paths = []
    for wav, label, name in train_dataset_v3:
        train_paths.append(name)
    train_paths.sort()

    valid_paths = []
    for wav, label, name in valid_dataset_v3:
        valid_paths.append(name)
    valid_paths.sort()

    test_paths = []
    for wav, label, name in test_dataset_v3:
        test_paths.append(name)
    test_paths.sort()

    from s3prl.base import fileio

    fileio.save("result/train", train_paths, "txt")
    fileio.save("result/valid", valid_paths, "txt")
    fileio.save("result/test", test_paths, "txt")

    cfg = SuperbER.setup.default_cfg
    train_data, valid_data, test_data = cfg.corpus(IEMOCAP, test_fold=fold_id).slice(3)
    train_dataset_v4 = cfg.train_datapipe["0"]()(train_data)
    valid_dataset_v4 = cfg.valid_datapipe["0"]()(valid_data, **train_dataset_v4.all_tools())
    test_dataset_v4 = cfg.test_datapipe["0"]()(test_data, **train_dataset_v4.all_tools())

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
