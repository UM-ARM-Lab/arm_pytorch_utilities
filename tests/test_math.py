import math

import torch
from arm_pytorch_utilities import math_utils


def test_angle_normalize():
    N = 100
    theta = torch.randn(N) * 5
    t1 = math_utils.angle_normalize(theta)
    assert torch.allclose(torch.cos(theta), torch.cos(t1), atol=1e-6)
    assert torch.allclose(torch.sin(theta), torch.sin(t1), atol=1e-6)


def test_batch_angle_rotate():
    N = 100
    xy = torch.tensor([1., 0]).repeat(N, 1)
    theta = (torch.rand(N) - 0.5) * 2 * math.pi

    xyr = math_utils.batch_rotate_wrt_origin(xy, theta)
    for i in range(N):
        r = math_utils.rotate_wrt_origin(xy[i], theta[i])
        assert torch.allclose(torch.tensor(r), xyr[i])


def test_cos_sim_pairwise():
    N = 100
    M = 30
    nx = 5
    x1 = torch.rand((N, nx))
    x2 = torch.rand((M, nx))
    C = math_utils.cos_sim_pairwise(x1, x2)
    assert C.shape == (N, M)
    for m in range(M):
        c = torch.cosine_similarity(x1, x2[m].view(1, -1))
        assert torch.allclose(c, C[:, m])


def test_angle_between():
    u = torch.tensor([[1., 0, 0]])
    v = torch.tensor([[0., 1, 0], [1, 0, 0]])
    assert torch.allclose(math_utils.angle_between(u, v), torch.tensor([[math.pi / 2, 0]]))

    u = torch.tensor([[1., 0, 0], [-1, 0, 0]])
    res = math_utils.angle_between(u, v)
    assert res.shape == (2, 2)
    assert torch.allclose(res, torch.tensor([[math.pi / 2, 0], [math.pi / 2, math.pi]]))

    N = 100
    M = 150
    u = torch.randn(N, 3)
    v = torch.randn(M, 3)

    res = math_utils.angle_between(u, v)
    res2 = math_utils.angle_between_stable(u, v)

    U = (u / u.norm(dim=-1, keepdim=True)).unsqueeze(1).repeat(1, M, 1)
    V = (v / v.norm(dim=-1, keepdim=True)).unsqueeze(0).repeat(N, 1, 1)
    close_to_parallel = torch.isclose(U, V, atol=2e-2) | torch.isclose(U, -V, atol=2e-2)
    close_to_parallel = close_to_parallel.all(dim=-1)
    # they should be the same when they are not close to parallel
    assert torch.allclose(res[~close_to_parallel],
                          res2[~close_to_parallel],
                          atol=1e-5)  # only time when they shouldn't be equal is when u ~= v or u ~= -v


def test_angle_between_batch():
    N = 100
    u = torch.randn(N, 3)
    res = math_utils.angle_between_batch(u, u)
    assert torch.allclose(res, torch.zeros(N))
    res = math_utils.angle_between_batch(u, -u)
    assert torch.allclose(res, math.pi * torch.ones(N))

    u = torch.randn(N, 2)
    # project onto 3d with z=0
    u = torch.cat((u, torch.zeros(N, 1)), dim=1)

    # rotate by 90 degrees around z
    R = torch.tensor([[0, 1, 0], [-1, 0, 0], [0, 0, 1]], dtype=u.dtype)
    v = u @ R
    res = math_utils.angle_between_batch(u, v)
    assert torch.allclose(res, math.pi / 2 * torch.ones(N))


def test_replace_nan_and_inf():
    # 1D tensor
    a = torch.tensor([1.0, float('nan'), 3.0, float('inf'), -float('inf'), 6.0])
    result = math_utils.replace_nan_and_inf(a.clone(), replacement=0)
    expected = torch.tensor([1.0, 0.0, 3.0, 0.0, 0.0, 6.0])
    assert torch.allclose(result, expected)

    # 2D tensor with custom replacement
    b = torch.tensor([[1.0, float('nan')], [float('inf'), 4.0]])
    result = math_utils.replace_nan_and_inf(b.clone(), replacement=5)
    expected = torch.tensor([[1.0, 5.0], [5.0, 4.0]])
    assert torch.allclose(result, expected)

    # Clean tensor passes through unchanged
    c = torch.tensor([1.0, 2.0, 3.0])
    result = math_utils.replace_nan_and_inf(c.clone())
    assert torch.allclose(result, c)


def test_clip():
    # Per-element tensor bounds (the differentiator from torch.clamp)
    a = torch.tensor([1.0, 5.0, -3.0, 7.0])
    min_val = torch.tensor([0.0, 2.0, -1.0, 0.0])
    max_val = torch.tensor([2.0, 4.0, 0.0, 6.0])
    result = math_utils.clip(a, min_val, max_val)
    expected = torch.tensor([1.0, 4.0, -1.0, 6.0])
    assert torch.allclose(result, expected)

    # Broadcasting: a is (N, D), bounds are (D,)
    N, D = 10, 3
    a = torch.randn(N, D) * 5
    min_val = torch.tensor([-1.0, -2.0, -3.0])
    max_val = torch.tensor([1.0, 2.0, 3.0])
    result = math_utils.clip(a, min_val, max_val)
    assert (result >= min_val).all()
    assert (result <= max_val).all()


def test_angular_diff_batch():
    # Known wrapping values
    a = torch.tensor([3.0, -3.0, 0.1])
    b = torch.tensor([0.1, 0.1, 3.0])
    result = math_utils.angular_diff_batch(a, b)
    # All results should be in (-pi, pi]
    assert (result > -math.pi).all()
    assert (result <= math.pi + 1e-6).all()

    # Compare against element-wise angular_diff with inputs where |a-b| < 2*pi
    # (scalar angular_diff only wraps once, so it's only correct in that range)
    N = 50
    a = (torch.rand(N) - 0.5) * 2 * math.pi
    b = (torch.rand(N) - 0.5) * 2 * math.pi
    batch_result = math_utils.angular_diff_batch(a, b)
    for i in range(N):
        scalar_result = math_utils.angular_diff(a[i].item(), b[i].item())
        assert abs(batch_result[i].item() - scalar_result) < 1e-5

    # Verify batch version handles large differences correctly (beyond single-wrap range)
    a_large = torch.tensor([10.0, -10.0, 20.0])
    b_large = torch.tensor([0.0, 0.0, 0.0])
    result_large = math_utils.angular_diff_batch(a_large, b_large)
    assert (result_large > -math.pi).all()
    assert (result_large <= math.pi + 1e-6).all()


def test_get_bounds():
    assert math_utils.get_bounds(None, 5) == (-5, 5)
    assert math_utils.get_bounds(-3, None) == (-3, 3)
    assert math_utils.get_bounds(2, 5) == (2, 5)
    assert math_utils.get_bounds(None, None) == (None, None)


if __name__ == "__main__":
    test_angle_normalize()
    test_batch_angle_rotate()
    test_cos_sim_pairwise()
    test_angle_between()
    test_angle_between_batch()
    test_replace_nan_and_inf()
    test_clip()
    test_angular_diff_batch()
    test_get_bounds()
