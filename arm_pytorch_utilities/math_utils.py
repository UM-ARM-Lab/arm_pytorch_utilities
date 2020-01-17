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
