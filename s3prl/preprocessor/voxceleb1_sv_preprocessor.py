import os
import logging
from tqdm import tqdm
from pathlib import Path

from s3prl.util.loader import TorchaudioLoader
from s3prl import Object, Output, cache

SPLIT_FILE_URL = "https://www.robots.ox.ac.uk/~vgg/data/voxceleb/meta/iden_split.txt"
TRIAL_FILE_URL = "https://openslr.magicdatatech.com/resources/49/voxceleb1_test_v2.txt"

class VoxCeleb1SVPreprocessor(Object):
    def __init__(self, dataset_root):
        super().__init__()
        self.speaker_num = 1211

        dataset_root = Path(dataset_root).resolve()

        split_filename = SPLIT_FILE_URL.split("/")[-1]
        if not (dataset_root / split_filename).is_file():
            os.system(f"wget {SPLIT_FILE_URL} -O {str(dataset_root)}/{split_filename}")

        trial_filename = TRIAL_FILE_URL.split("/")[-1]
        if not (dataset_root / trial_filename).is_file():
            os.system(f"wget {TRIAL_FILE_URL} -O {str(dataset_root)}/{trial_filename}")

        self.train_path, self.valid_path, self.test_path, self.speakerid2label = self.standard_split(
            dataset_root, split_filename
        )
        self.train_label = self.build_label(self.train_path)
        self.valid_label = self.build_label(self.valid_path)
        self.test_label  = self.format_test_trials(dataset_root, trial_filename)

        categories = list(set([*self.train_label, *self.valid_label]))

        assert len(self.speakerid2label) == self.speaker_num

    def build_label(self, train_path_list):
        y = []
        for path in train_path_list:
            id_string = path.split("/")[-3]
            y.append(self.speakerid2label[id_string])
        return y

    @staticmethod
    @cache()
    def format_test_trials(dataset_root, trial_filename):
        meta_data   = dataset_root / trial_filename
        usage_list  = open(meta_data, "r").readlines()

        test_trials = []
        prefix      = dataset_root / "test/wav"

        for string in tqdm(usage_list, desc="Prepare testing trials"):
            pair = string.split()
            test_trials.append((int(pair[0]), str(prefix / pair[1]), str(prefix / pair[2])))
        
        return test_trials

    @staticmethod
    @cache()
    def standard_split(dataset_root, split_filename):
        meta_data  = dataset_root / split_filename
        usage_list = open(meta_data, "r").readlines()

        train, valid, test = [], [], []
        test_list  = [item for item in usage_list if int(item.split(' ')[1].split('/')[0][2:]) in range(10270, 10310)]
        usage_list = list(set(usage_list).difference(set(test_list)))
        test_list  = [item.split(" ")[1] for item in test_list]
        
        logging.info("search specified wav name for each split")
        speakerids = []
        
        for string in tqdm(usage_list, desc="Search train, dev wavs"):
            pair  = string.split()
            index = pair[0]
            x     = list(dataset_root.glob("dev/wav/" + pair[1]))
            speakerStr = pair[1].split('/')[0]
            if speakerStr not in speakerids:
                speakerids.append(speakerStr)
            if int(index) == 1 or int(index) == 3:
                train.append(str(x[0]))
            elif int(index) == 2:
                valid.append(str(x[0]))
            else:
                raise ValueError
        
        speakerids = sorted(speakerids)
        speakerid2label = {}
        for idx, spk in enumerate(speakerids):
            speakerid2label[spk] = idx
        
        for string in tqdm(test_list, desc="Search test wavs"):
            x = list(dataset_root.glob("test/wav/" + string.strip()))
            test.append(str(x[0]))
        logging.info(
            f"finish searching wav: train {len(train)}; valid {len(valid)}; test {len(test)} files found"
        )
        return train, valid, test, speakerid2label

    def train_data(self):
        return Output(
            source=self.train_path,
            label=self.train_label,
            category=self.speakerid2label,
            source_loader=TorchaudioLoader(),
        )

    def valid_data(self):
        return Output(
            source=self.valid_path,
            label=self.valid_label,
            category=self.speakerid2label,
            source_loader=TorchaudioLoader(),
        )

    def test_data(self):
        return Output(
            source=self.test_path,
            label=self.test_label,
            source_loader=TorchaudioLoader(),
        )

    def statistics(self):
        return Output(input_size=1, category=self.speakerid2label)

if __name__ == "__main__":
    dataset_root = "/mnt/ssd-201-112-01/cpii.local/bzheng/voxceleb1"
    preprocessor = VoxCeleb1SVPreprocessor(dataset_root)