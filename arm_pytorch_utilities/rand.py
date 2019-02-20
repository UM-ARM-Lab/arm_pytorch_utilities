import random
import torch


def seed(randseed=None):
    """Seed RNG (CPU and CUDA) with either a given seed or a generated one and return the seed"""
    if randseed is None:
        randseed = random.randint(0, 1000000)

    torch.manual_seed(randseed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(randseed)

    return randseed
