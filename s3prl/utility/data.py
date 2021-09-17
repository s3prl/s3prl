from torch.distributed.distributed_c10d import is_initialized
from torch.utils.data import Dataset, DistributedSampler

def get_ddp_sampler(dataset: Dataset, epoch: int):
    """
    This function will create a DistributedSampler if DDP is initialized,
    and will just return None if DDP is not initialized.
    """
    if is_initialized():
        sampler = DistributedSampler(dataset)
        sampler.set_epoch(epoch)
    else:
        sampler = None
    return sampler
