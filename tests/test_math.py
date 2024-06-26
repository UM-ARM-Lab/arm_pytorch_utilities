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


if __name__ == "__main__":
    test_angle_normalize()
    test_batch_angle_rotate()
    test_cos_sim_pairwise()
    test_angle_between()
    test_angle_between_batch()
