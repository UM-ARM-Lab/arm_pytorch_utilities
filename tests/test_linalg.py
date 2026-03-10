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


def test_batch_quadratic_product():
    N = 100
    nx = 10
    # Create PSD matrix: A = R^T R
    R = torch.randn(nx, nx)
    A = R.t() @ R
    X = torch.randn(N, nx)

    result = linalg.batch_quadratic_product(X, A)
    assert result.shape == (N,)
    for i in range(N):
        expected = X[i] @ A @ X[i]
        assert torch.allclose(result[i], expected, atol=1e-4)

    # Identity matrix should give squared norms
    I = torch.eye(nx)
    result = linalg.batch_quadratic_product(X, I)
    expected = (X * X).sum(dim=1)
    assert torch.allclose(result, expected, atol=1e-5)


def test_kronecker_product():
    # Compare against torch.kron for random matrices
    A = torch.randn(3, 4)
    B = torch.randn(2, 5)
    result = linalg.kronecker_product(A, B)
    expected = torch.kron(A, B)
    assert torch.allclose(result, expected, atol=1e-6)

    # Known value: kron(I_2, I_3) = I_6
    I2 = torch.eye(2)
    I3 = torch.eye(3)
    result = linalg.kronecker_product(I2, I3)
    assert torch.allclose(result, torch.eye(6))


if __name__ == "__main__":
    test_cov()
    test_sqrtm()
    test_batch_outer_prodcut()
    test_batch_batch_product()
    test_batch_quadratic_product()
    test_kronecker_product()
