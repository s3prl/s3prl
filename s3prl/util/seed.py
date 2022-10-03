"""
Fix random seeds

Authors
  * Heng-Jui Chang 2022
"""

import random

import numpy as np
import torch

__all__ = [
    "fix_random_seeds",
]


def fix_random_seeds(seed: int = 1337) -> None:
    """Fixes all random seeds, including cuDNN backends.

    Args:
        seed (int, optional): Random seed. Defaults to 1337.
    """

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
