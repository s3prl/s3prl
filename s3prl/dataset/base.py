import abc

from torch.utils.data import Dataset, DataLoader

from s3prl import Object


class Dataset(Object, Dataset):
    def __init__(self):
        super().__init__()

    @abc.abstractmethod
    def prepare_data(self):
        pass

    @abc.abstractmethod
    def collate_fn(self):
        pass

    def to_dataloader(
        self,
        batch_size=1,
        shuffle=False,
        sampler=None,
        batch_sampler=None,
        num_workers=0,
        collate_fn=None,
        pin_memory=False,
        drop_last=False,
        timeout=0,
        worker_init_fn=None,
        prefetch_factor=2,
        persistent_workers=False,
    ) -> DataLoader:
        """
        if collate_fn is None, use self.collate_fn,
        all other arguments are untouched for torch.utils.data.DataLoader
        """
        dataloader = DataLoader(
            self,
            batch_size=batch_size,
            shuffle=shuffle,
            sampler=sampler,
            batch_sampler=batch_sampler,
            num_workers=num_workers,
            collate_fn=collate_fn or self.collate_fn,
            pin_memory=pin_memory,
            drop_last=drop_last,
            timeout=timeout,
            worker_init_fn=worker_init_fn,
            prefetch_factor=prefetch_factor,
            persistent_workers=persistent_workers,
        )
        return dataloader
