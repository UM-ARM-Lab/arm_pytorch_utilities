import torch
from arm_pytorch_utilities import rand


def test_seed():
    N = 100
    rand.seed(5)
    r1 = torch.randn(N)
    rand.seed(5)
    r2 = torch.randn(N)
    assert torch.allclose(r1, r2)


def test_save_rng():
    N = 100
    rand.seed(5)
    r1 = torch.randn(N)

    rand.seed(5)
    with rand.SavedRNG():
        torch.randn(N)
    r2 = torch.randn(N)
    assert torch.allclose(r1, r2)


if __name__ == "__main__":
    test_seed()
    test_save_rng()
