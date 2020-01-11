from itertools import groupby

import numpy as np
import torch


def discrete_array_to_value_ranges(discrete_array):
    """Return a sequence of (value, start, end) for an array of discrete values that have substrings"""
    i = 0
    for c, g in groupby(discrete_array):
        frames = sum(1 for _ in g)
        yield c, i, i + frames
        i += frames


def check_shape(value, expected_shape, name=''):
    """
    Throws a ValueError if value.shape != expected_shape.
    Args:
        value: Matrix to shape check.
        expected_shape: A tuple or list of integers.
        name: An optional name to add to the exception message.
    """
    if value.shape != tuple(expected_shape):
        raise ValueError('Shape mismatch %s: Expected %s, got %s' %
                         (name, str(expected_shape), str(value.shape)))


def extract_positive_weights(weights):
    """
    Extract the indices and values of weights that are positive
    :param weights: single dimensional weight tensor, or non-floating point array/np.ndarray
    :return: indices, values, number
    """
    # assume floating point type means weights are soft
    if torch.is_tensor(weights) and torch.is_floating_point(weights):
        return _extract_soft_weights(weights)
    return _extract_hard_weights(weights)


def _extract_soft_weights(w):
    neighbours = w > 0
    nw = w[neighbours]
    return neighbours, nw, nw.shape[0]


def _extract_hard_weights(w):
    if torch.is_tensor(w):
        return w, torch.ones_like(w).double()
    return w, np.ones_like(w), len(w)
