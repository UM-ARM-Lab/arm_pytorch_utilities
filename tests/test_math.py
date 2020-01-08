import torch
from arm_pytorch_utilities import math_utils


def test_angle_normalize():
    N = 100
    theta = torch.randn(N) * 5
    t1 = math_utils.angle_normalize(theta)
    assert torch.allclose(torch.cos(theta), torch.cos(t1),  atol=1e-6)
    assert torch.allclose(torch.sin(theta), torch.sin(t1),  atol=1e-6)


if __name__ == "__main__":
    test_angle_normalize()
