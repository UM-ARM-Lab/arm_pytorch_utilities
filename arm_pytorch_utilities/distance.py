import torch


def pairwise_distance(x, y=None):
    """Euclidean pairwise distance"""
    if y is None:
        y = x.t()

    y = x.t()
    out = -2 * torch.matmul(x, y)
    out += (x ** 2).sum(dim=-1, keepdim=True)
    out += (y ** 2).sum(dim=-2, keepdim=True)

    return out
