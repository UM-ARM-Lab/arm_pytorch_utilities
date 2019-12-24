from itertools import groupby


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