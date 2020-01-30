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


if __name__ == "__main__":
    test_ensure_diagonal()
