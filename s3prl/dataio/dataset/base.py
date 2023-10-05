from torch.utils import data


class Dataset(data.Dataset):
    def __len__(self) -> int:
        raise NotImplementedError

    def __getitem__(self, index: int):
        raise NotImplementedError

    def getinfo(self, index: int):
        raise NotImplementedError
