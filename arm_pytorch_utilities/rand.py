import random
import torch
import numpy as np


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


class SavedRNG:
    """Save RNG state before entering context and restore it after

    Useful for debugging sections for preventing change of behaviour while debugging that involves the RNG
    Example usage:
    rand.seed(5)
    with rand.SavedRNG():
        torch.randn(N)
    # would still be seeded 5 here
    """

    def __init__(self):
        self.np_state = None
        self.rng_state = None
        self.gpu_state = None
        self.save_gpu = torch.cuda.is_available()

    def __enter__(self):
        self.np_state = np.random.get_state()
        self.rng_state = torch.get_rng_state()
        if self.save_gpu:
            self.gpu_state = torch.cuda.get_rng_state()

    def __exit__(self, *args):
        np.random.set_state(self.np_state)
        torch.set_rng_state(self.rng_state)
        if self.save_gpu:
            torch.cuda.set_rng_state(self.gpu_state)
