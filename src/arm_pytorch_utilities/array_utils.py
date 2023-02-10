import re
from itertools import groupby

import torch


def discrete_array_to_value_ranges(discrete_array):
    """
    Return a sequence of (value, start, end) for an array of discrete values that have substrings

    discrete_array[start] up to and including discrete_array[end] has the same value
    """
    i = 0
    for c, g in groupby(discrete_array):
        frames = sum(1 for _ in g)
        yield c, i, i + frames - 1
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


def extract_positive_weights(weights, force_numerical_weights=False):
    """
    Extract the indices and values of weights that are positive.
    :param weights: single dimensional weight tensor, or non-floating point array/np.ndarray
    :param force_numerical_weights: whether the weights have to be numerical, or if they can be None on hard weights
    :return: indices, values, number
    """
    # assume floating point type means weights are soft
    if torch.is_tensor(weights) and torch.is_floating_point(weights):
        return _extract_soft_weights(weights)
    indices, weights, N = _extract_hard_weights(weights)
    if force_numerical_weights:
        weights = torch.ones(N, dtype=torch.double)
    return indices, weights, N


def _extract_soft_weights(w):
    neighbours = w > 0
    nw = w[neighbours]
    return neighbours, nw, nw.shape[0]


def _extract_hard_weights(w):
    if torch.is_tensor(w):
        return w, None, w.shape[0]
    if isinstance(w, slice):
        N = w.stop - w.start
        return w, None, N
    else:
        raise RuntimeError("Unhandled weight type {}".format(type(w)))


def _tryint(s):
    try:
        return int(s)
    except:
        return s


def _alphanum_key(s):
    """
    Turn a string into a list of string and number chunks.
        "z23a" -> ["z", 23, "a"]
    :param s:
    :return:
    """
    return [_tryint(c) for c in re.split('([0-9]+)', s)]


def sort_nicely(l):
    """
    In place sort list in the way that humans expects; from https://nedbatchelder.com/blog/200712/human_sorting.html
    :param l: list
    :return:
    """
    l.sort(key=_alphanum_key)
