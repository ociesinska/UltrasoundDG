import random

import numpy as np
import torch


def set_random_seed(seed: int) -> None:
    """Seed Python, NumPy, PyTorch, and every available CUDA device."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
