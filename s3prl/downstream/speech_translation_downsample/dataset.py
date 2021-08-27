from torch.utils.data import Dataset

class DummyDataset(Dataset):

    def __init__(self, dataset, batch_sampler):

        self.dataset = dataset
        self.batch_sampler = batch_sampler

    def __len__(self):

        return len(self.batch_sampler)

    def __getitem__(self, idx):
        
        batch = [self.dataset[sample_id] for sample_id in self.batch_sampler[idx]]
        return self.dataset.collater(batch)

    def collate_fn(self, samples):

        assert len(samples) == 1
        return samples[0]