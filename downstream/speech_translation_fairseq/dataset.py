from torch.utils.data import Dataset

class DummyDataset(Dataset):

    def __init__(self, dataset, batch_sampler, num_workers=1):

        self.dataset = dataset
        self.batch_sampler = batch_sampler
        self.num_workers = num_workers

    def __len__(self):

        return len(self.batch_sampler)

    def __getitem__(self, idx):

        batch = []

        for sample_id in self.batch_sampler[idx]:
            batch.append(self.dataset[sample_id])
        
        return self.dataset.collater(batch)

    def collate_fn(self, samples):

        assert len(samples) == 1

        return samples[0]