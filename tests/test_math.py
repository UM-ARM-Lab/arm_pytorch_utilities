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


if __name__ == "__main__":
    test_angle_normalize()
    test_batch_angle_rotate()
