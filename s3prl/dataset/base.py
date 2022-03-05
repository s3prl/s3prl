import abc

from torch.utils.data import Dataset

from s3prl import Object, init

class Dataset(Object, Dataset):
    def __init__(self):
        super().__init__()

    @abc.abstractmethod
    def prepare_data(self):
        pass

    @abc.abstractmethod
    def collate_fn(self):
        pass
