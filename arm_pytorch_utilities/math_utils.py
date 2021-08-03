import math

import torch
from arm_pytorch_utilities import linalg


def clip(a, min_val, max_val):
    """Vectorized torch.clamp (supports tensors for min_val and max_val)"""
    return torch.max(torch.min(a, max_val), min_val)


def replace_nan_and_inf(a, replacement=0):
    """Replaces nan,inf,-inf values with replacement value in place"""
    a[torch.isnan(a)] = replacement
    a[a == float('inf')] = replacement
    a[a == -float('inf')] = replacement
    return a


def get_bounds(u_min=None, u_max=None):
    """Convenience for ensuring a bound if given is two sided"""
    # make sure if any of them is specified, both are specified
    if u_max is not None and u_min is None:
        u_min = -u_max
    if u_min is not None and u_max is None:
        u_max = -u_min
    return u_min, u_max


def rotate_wrt_origin(xy, theta):
    return (xy[0] * math.cos(theta) - xy[1] * math.sin(theta),
            xy[0] * math.sin(theta) + xy[1] * math.cos(theta))


def batch_rotate_wrt_origin(xy, theta):
    # create rotation matrices
    c = torch.cos(theta).view(-1, 1)
    s = torch.sin(theta).view(-1, 1)
    top = torch.cat((c, s), dim=1)
    bot = torch.cat((-s, c), dim=1)
    R = torch.cat((top.unsqueeze(2), bot.unsqueeze(2)), dim=2)
    return linalg.batch_batch_product(xy, R)


def angle_between(u: torch.tensor, v: torch.tensor):
    """Angle between 2 n-dimensional vectors

    :param u N x n tensor
    :param v M x n tensor
    :return N x M angle in radians between each tensor
    """
    u_n = u / u.norm(dim=1, keepdim=True)
    v_n = v / v.norm(dim=1, keepdim=True)
    c = clip(u_n @ v_n.transpose(0, 1),
             torch.tensor(-1, device=u.device, dtype=u.dtype), torch.tensor(1, device=u.device, dtype=u.dtype))
    return torch.acos(c)


def angular_diff(a, b):
    """Angle difference from b to a (a - b)"""
    d = a - b
    if d > math.pi:
        d -= 2 * math.pi
    elif d < -math.pi:
        d += 2 * math.pi
    return d


def angular_diff_batch(a, b):
    """Angle difference from b to a (a - b)"""
    d = a - b
    d[d > math.pi] -= 2 * math.pi
    d[d < -math.pi] += 2 * math.pi
    return d


def angle_normalize(a):
    """Wrap angle between (-pi,pi)"""
    return ((a + math.pi) % (2 * math.pi)) - math.pi


def cos_sim_pairwise(x1, x2=None, eps=1e-8):
    """Pairwise cosine similarity between 2 sets of vectors

    From https://github.com/pytorch/pytorch/issues/11202
    :param x1: (N, nx) N tensors of nx dimension each
    :param x2: (M, nx) M tensors of nx dimension each, equal to x1 if not specified
    :param eps: threshold above 0 to reduce dividing by 0
    :return: (N, M) pairwise cosine similarity
    """
    x2 = x1 if x2 is None else x2
    w1 = x1.norm(p=2, dim=1, keepdim=True)
    w2 = w1 if x2 is x1 else x2.norm(p=2, dim=1, keepdim=True)
    return torch.mm(x1, x2.t()) / (w1 * w2.t()).clamp(min=eps)
