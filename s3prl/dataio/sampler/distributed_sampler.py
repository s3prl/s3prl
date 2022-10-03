"""
Wrap any batch sampler for distributed training

Authors:
  * Leo 2022
"""

import logging
from copy import deepcopy
from typing import Iterator, Optional, TypeVar

import torch.distributed as dist
from torch.utils.data import BatchSampler

T_co = TypeVar("T_co", covariant=True)
logger = logging.getLogger(__name__)

__all__ = [
    "DistributedBatchSamplerWrapper",
]


class DistributedBatchSamplerWrapper:
    def __init__(
        self,
        batch_sampler: BatchSampler,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        allow_duplicates: bool = False,
        allow_uneven: bool = False,
    ) -> None:
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        if rank >= num_replicas or rank < 0:
            raise ValueError(
                "Invalid rank {}, rank should be in the interval"
                " [0, {}]".format(rank, num_replicas - 1)
            )
        self.batch_sampler = batch_sampler
        self.num_replicas = num_replicas
        self.rank = rank
        self.allow_duplicates = allow_duplicates
        self.allow_uneven = allow_uneven

    def __iter__(self) -> Iterator[T_co]:
        logger.info(
            f"Building distributed batch sampler for rank={self.rank}, world_size={self.num_replicas}"
        )

        all_rank_batch_indices = list(iter(self.batch_sampler))
        if len(all_rank_batch_indices) % self.num_replicas == 0:
            target_batch_indices = all_rank_batch_indices
        else:
            num_to_halve = (
                self.num_replicas - len(all_rank_batch_indices) % self.num_replicas
            )
            flatten_batch_indices = deepcopy(all_rank_batch_indices)
            while num_to_halve > 0:
                newly_flatten = []
                all_cant_be_halved = True
                for indices in flatten_batch_indices:
                    if num_to_halve > 0 and len(indices) > 1:
                        indices1, indices2 = (
                            indices[: len(indices) // 2],
                            indices[len(indices) // 2 :],
                        )
                        newly_flatten += [indices1, indices2]
                        num_to_halve -= 1
                        all_cant_be_halved = False
                    else:
                        newly_flatten.append(indices)
                flatten_batch_indices = deepcopy(newly_flatten)

                if all_cant_be_halved:
                    if self.allow_duplicates:
                        logger.warning(
                            "To ensure all the dataloaders in different processes get the same number "
                            "of batches. Some batches are duplicated. This must not happen during the "
                            "evaluation stage."
                        )
                        flatten_batch_indices = (
                            flatten_batch_indices
                            + all_rank_batch_indices[:num_to_halve]
                        )
                    elif self.allow_uneven:
                        logger.warning(
                            "Total batches will not be evenly distributed across the dataloaders in "
                            "different processes. This must not happen during the training stage and "
                            "can lead to hanging, while might be okay during the evaluation stage."
                        )
                    else:
                        raise ValueError(
                            "The provided batch sampler cannot be safely wrapped for distributed training. "
                            "Please try increase the number of indices in each batch. Or, allowing duplicated "
                            "batches or uneven number of batches across dataloaders."
                        )
            target_batch_indices = flatten_batch_indices

        if not self.allow_uneven:
            assert len(target_batch_indices) % self.num_replicas == 0

        batch_indices = target_batch_indices[self.rank :: self.num_replicas]
        return iter(batch_indices)

    def __len__(self) -> int:
        # Since the total number of batches dynamically depends on the current epoch,
        # instead of pre-compute it which will duplicate the batch number computation logic,
        # it makes no harm to simply re-compute it with __iter__ for every call, since
        # __len__ is usually not frequently called and won't be the performance bottleneck
        return len(list(iter(self)))

    def set_epoch(self, epoch: int) -> None:
        if hasattr(self.batch_sampler, "set_epoch"):
            self.batch_sampler.set_epoch(epoch)
