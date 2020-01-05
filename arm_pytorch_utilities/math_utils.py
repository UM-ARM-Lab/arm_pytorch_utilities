import math
import torch


def clip(a, min_val, max_val):
    """Vectorized torch.clamp (supports tensors for min_val and max_val)"""
    return torch.max(torch.min(a, max_val), min_val)


def rotate_wrt_origin(xy, theta):
    return (xy[0] * math.cos(theta) + xy[1] * math.sin(theta),
            -xy[0] * math.sin(theta) + xy[1] * math.cos(theta))


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
