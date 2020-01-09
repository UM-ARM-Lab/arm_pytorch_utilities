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


def test_sqrtm():
    from torch.autograd import gradcheck
    k = torch.randn(20, 10).double()
    # Create a positive definite matrix
    pd_mat = k.t().matmul(k)
    pd_mat.requires_grad = True
    test = gradcheck(linalg.sqrtm, (pd_mat,))
    assert test is True


def test_batch_outer_prodcut():
    n = 5
    N = 10
    u = torch.rand(N, n)
    v = torch.rand(N, n)
    UV = linalg.batch_outer_product(u, v)
    assert UV.shape == (N, n, n)
    for i in range(N):
        uv = torch.ger(u[i], v[i])
        assert torch.allclose(uv, UV[i])


def test_batch_batch_product():
    B = 1000
    ny = 10
    nx = 20
    A = torch.rand(B, ny, nx)
    X = torch.rand(B, nx)

    Y = linalg.batch_batch_product(X, A)
    assert Y.shape == (B, ny)
    for i in range(B):
        y = A[i] @ X[i]
        assert torch.allclose(Y[i], y)


if __name__ == "__main__":
    test_cov()
    test_sqrtm()
    test_batch_outer_prodcut()
    test_batch_batch_product()
