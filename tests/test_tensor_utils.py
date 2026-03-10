import torch
from arm_pytorch_utilities import tensor_utils


def test_ensure_diagonal():
    nx = 5
    q = torch.rand(nx)
    Q = torch.diag(q)
    Q1 = tensor_utils.ensure_diagonal(Q, nx)
    assert torch.allclose(Q, Q1)
    Q2 = tensor_utils.ensure_diagonal(q, nx)
    assert torch.allclose(Q, Q2)
    Q3 = tensor_utils.ensure_diagonal(q.numpy(), nx)
    assert torch.allclose(Q, Q3)
    Q4 = tensor_utils.ensure_diagonal(Q.numpy(), nx)
    assert torch.allclose(Q, Q4)

    q = torch.rand(1).item()
    Q = torch.eye(nx) * q
    Q1 = tensor_utils.ensure_diagonal(q, nx)
    assert torch.allclose(Q, Q1)


def test_handle_batch_input():
    @tensor_utils.handle_batch_input(n=2)
    def add_and_average(a, b):
        assert len(a.shape) == 2
        return a + b, (a + b).mean()

    @tensor_utils.handle_batch_input(n=3)
    def add_and_average3(a, b):
        assert len(a.shape) == 3
        return a + b, (a + b).mean()

    B = 4
    N = 10
    nx = 3
    A = torch.rand((B, N, nx))
    Ahat, mean = add_and_average(A, 0)

    assert A.shape == Ahat.shape
    assert torch.allclose(A, Ahat)

    Ahat, mean = add_and_average3(A, 0)
    assert A.shape == Ahat.shape
    assert torch.allclose(A, Ahat)


def test_squeeze_n():
    v = torch.randn(1, 1, 1, 5, 3)
    # n=2: squeeze 2 leading dims
    r = tensor_utils.squeeze_n(v, 2)
    assert r.shape == (1, 5, 3)
    # n=0: unchanged
    r = tensor_utils.squeeze_n(v, 0)
    assert r.shape == (1, 1, 1, 5, 3)
    # n=3: squeeze 3 leading dims
    r = tensor_utils.squeeze_n(v, 3)
    assert r.shape == (5, 3)


def test_first_positive():
    x = torch.tensor([[-1., -2., 3., 4.], [5., -1., 2., 0.]])
    values, indices = tensor_utils.first_positive(x, dim=1)
    assert indices[0] == 2
    assert indices[1] == 0

    # All positive: should return index 0
    x = torch.tensor([[1., 2., 3.]])
    values, indices = tensor_utils.first_positive(x, dim=1)
    assert indices[0] == 0

    # No positive values
    x = torch.tensor([[-1., -2., -3.]])
    values, indices = tensor_utils.first_positive(x, dim=1)
    # When no positive values, cumsum never reaches 1 while nonz is True
    # so max returns 0 (False) with index 0
    assert values[0] == 0


def test_ensure_tensor():
    device = torch.device('cpu')
    dtype = torch.float32

    # Single arg -> single tensor (not tuple)
    result = tensor_utils.ensure_tensor(device, dtype, [1.0, 2.0, 3.0])
    assert torch.is_tensor(result)
    assert result.shape == (3,)

    # Numpy array
    import numpy as np
    result = tensor_utils.ensure_tensor(device, dtype, np.array([4.0, 5.0]))
    assert torch.is_tensor(result)

    # Existing tensor
    t = torch.tensor([1.0, 2.0])
    result = tensor_utils.ensure_tensor(device, dtype, t)
    assert torch.is_tensor(result)
    assert torch.allclose(result, t)

    # Multiple args -> tuple
    r1, r2 = tensor_utils.ensure_tensor(device, dtype, [1.0], [2.0])
    assert torch.is_tensor(r1)
    assert torch.is_tensor(r2)


def test_handle_batch_input_extra_dims():
    @tensor_utils.handle_batch_input(n=2)
    def identity_2d(x):
        assert len(x.shape) == 2
        return x

    # 1D input (under-dimension): should be expanded then squeezed back
    x_1d = torch.randn(5)
    result = identity_2d(x_1d)
    assert result.shape == x_1d.shape
    assert torch.allclose(result, x_1d)

    # 4D input (over-dimension): batch dims flattened then restored
    x_4d = torch.randn(2, 3, 4, 5)
    result = identity_2d(x_4d)
    assert result.shape == x_4d.shape
    assert torch.allclose(result, x_4d)

    # 3D with n=2 (one extra batch dim)
    x_3d = torch.randn(7, 4, 5)
    result = identity_2d(x_3d)
    assert result.shape == x_3d.shape
    assert torch.allclose(result, x_3d)


if __name__ == "__main__":
    test_ensure_diagonal()
    test_handle_batch_input()
    test_squeeze_n()
    test_first_positive()
    test_ensure_tensor()
    test_handle_batch_input_extra_dims()
