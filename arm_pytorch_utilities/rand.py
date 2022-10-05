import random
from typing import Optional

import numpy as np
import torch


def seed(randseed=None):
    """Seed RNG (CPU and CUDA) with either a given seed or a generated one and return the seed"""
    if randseed is None:
        randseed = random.randint(0, 1000000)

    random.seed(randseed)
    np.random.seed(randseed)
    torch.manual_seed(randseed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(randseed)

    return randseed


class RNGState:

    def __init__(self):
        self.np_state = None
        self.rng_state = None
        self.gpu_state = None
        self.save_gpu = torch.cuda.is_available()

    def __bool__(self):
        return self.rng_state is not None

    def restore(self):
        np.random.set_state(self.np_state)
        torch.set_rng_state(self.rng_state)
        if self.save_gpu:
            torch.cuda.set_rng_state(self.gpu_state)

    def save(self):
        self.np_state = np.random.get_state()
        self.rng_state = torch.get_rng_state()
        if self.save_gpu:
            self.gpu_state = torch.cuda.get_rng_state()


class SavedRNG:

    def __init__(self, initial_seed: Optional[int] = None):
        self.initial_seed = initial_seed
        self.local_rng_state = RNGState()
        self.global_rng_state = RNGState()

    def __enter__(self):
        self.global_rng_state.save()
        if not self.local_rng_state:
            if self.initial_seed is not None:
                seed(self.initial_seed)
            self.local_rng_state.save()

        self.local_rng_state.restore()

    def __exit__(self, *args):
        self.local_rng_state.save()
        self.global_rng_state.restore()
