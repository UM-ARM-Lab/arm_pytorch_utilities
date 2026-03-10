import torch
from arm_pytorch_utilities import softknn


def test_softknn_shapes():
    N = 50
    n = 3
    min_k = 5
    features = torch.randn(N, n)
    knn = softknn.SoftKNN(min_k=min_k)
    weights = knn(features)
    assert weights.shape == (N, N)
    assert (weights >= 0).all()


def test_softknn_gradient():
    N = 30
    n = 4
    features = torch.randn(N, n, requires_grad=True)
    knn = softknn.SoftKNN(min_k=5)
    weights = knn(features)
    weights.sum().backward()
    assert features.grad is not None
    assert features.grad.shape == (N, n)


def test_softknn_normalization():
    N = 20
    n = 3
    features = torch.randn(N, n)

    # L1 normalization: rows sum to 1
    knn1 = softknn.SoftKNN(min_k=5, normalization=1)
    w1 = knn1(features)
    row_sums = w1.sum(dim=1)
    assert torch.allclose(row_sums, torch.ones(N), atol=1e-5)

    # L2 normalization: rows have unit L2 norm
    knn2 = softknn.SoftKNN(min_k=5, normalization=2)
    w2 = knn2(features)
    row_norms = w2.norm(dim=1)
    assert torch.allclose(row_norms, torch.ones(N), atol=1e-5)


if __name__ == "__main__":
    test_softknn_shapes()
    test_softknn_gradient()
    test_softknn_normalization()
