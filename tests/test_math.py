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


if __name__ == "__main__":
    test_angle_normalize()
    test_batch_angle_rotate()
    test_cos_sim_pairwise()
    test_angle_between()
