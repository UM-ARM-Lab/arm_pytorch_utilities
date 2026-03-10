"""torch.compile verification tests.

These track which functions can be compiled with fullgraph=True.
Pre-refactor failures are marked xfail. Remove xfail as functions are refactored.
"""
import math

import pytest
import torch

from arm_pytorch_utilities import math_utils, linalg, tensor_utils, preprocess, softknn


def assert_compile_correct(fn, *args, atol=1e-5):
    eager = fn(*args)
    compiled_fn = torch.compile(fn, fullgraph=True)
    compiled = compiled_fn(*args)
    if isinstance(eager, tuple):
        for e, c in zip(eager, compiled):
            if torch.is_tensor(e):
                assert torch.allclose(e, c, atol=atol), f"Mismatch: eager {e} vs compiled {c}"
    else:
        assert torch.allclose(eager, compiled, atol=atol), f"Mismatch: eager {eager} vs compiled {compiled}"


def test_compile_replace_nan_and_inf():
    a = torch.tensor([1.0, float('nan'), 3.0, float('inf'), -float('inf'), 6.0])
    assert_compile_correct(math_utils.replace_nan_and_inf, a.clone(), 0)


def test_compile_angular_diff_batch():
    a = torch.tensor([3.0, -3.0, 0.1, 6.0])
    b = torch.tensor([0.1, 0.1, 3.0, -1.0])
    assert_compile_correct(math_utils.angular_diff_batch, a, b)


def test_compile_cos_sim_pairwise():
    x1 = torch.randn(20, 5)
    x2 = torch.randn(15, 5)
    assert_compile_correct(math_utils.cos_sim_pairwise, x1, x2)


def test_compile_angle_between_stable():
    u = torch.randn(10, 3)
    v = torch.randn(8, 3)
    assert_compile_correct(math_utils.angle_between_stable, u, v)


def test_compile_batch_quadratic_product():
    X = torch.randn(50, 5)
    R = torch.randn(5, 5)
    A = R.t() @ R
    assert_compile_correct(linalg.batch_quadratic_product, X, A)


def test_compile_first_positive():
    x = torch.tensor([[-1., -2., 3., 4.], [5., -1., 2., 0.]])

    def first_pos(t):
        return tensor_utils.first_positive(t, dim=1)

    assert_compile_correct(first_pos, x)


def test_compile_softknn_forward_linear():
    features = torch.randn(20, 5)
    knn = softknn.SoftKNN(min_k=5, activation='linear')
    assert_compile_correct(knn, features)


def test_compile_softknn_forward_sigmoid():
    features = torch.randn(20, 5)
    knn = softknn.SoftKNN(min_k=5, activation=10.0)
    assert_compile_correct(knn, features)


def test_compile_minmax_scaler_transform():
    scaler = preprocess.MinMaxScaler()
    x = torch.randn(100, 5)
    scaler.fit(x)
    assert_compile_correct(scaler.transform, x)


@pytest.mark.xfail(reason="pre-refactor: .numpy() in sqrtm autograd")
def test_compile_sqrtm():
    k = torch.randn(10, 5)
    A = k.t() @ k + torch.eye(5) * 0.1
    assert_compile_correct(linalg.sqrtm, A)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
