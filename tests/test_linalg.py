import torch
import numpy as np
from arm_pytorch_utilities import linalg


def assert_same_cov(A, w=None):
    c1 = np.cov(A, rowvar=False, aweights=w)
    c2 = linalg.cov(torch.tensor(A, dtype=torch.float), aweights=w)
    assert np.linalg.norm(c2.numpy() - c1) < 1e-6


def test_cov():
    a = [1, 2, 3, 4]
    assert_same_cov(a)
    A = [[1, 2], [3, 4]]
    assert_same_cov(A)

    assert_same_cov(a, w=[1, 1, 1, 1])
    assert_same_cov(a, w=[2, 0.5, 3, 1])

    assert_same_cov(A, w=[1, 1])
    assert_same_cov(A, w=[2, 0.5])


if __name__ == "__main__":
    test_cov()