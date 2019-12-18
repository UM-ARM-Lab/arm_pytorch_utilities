from itertools import groupby


def discrete_array_to_value_ranges(discrete_array):
    """Return a sequence of (value, start, end) for an array of discrete values that have substrings"""
    i = 0
    for c, g in groupby(discrete_array):
        frames = sum(1 for _ in g)
        yield c, i, i + frames
        i += frames
