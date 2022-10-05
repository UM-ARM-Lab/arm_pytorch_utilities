import torch
from arm_pytorch_utilities import rand


def test_seed():
    N = 100
    rand.seed(5)
    r1 = torch.randn(N)
    rand.seed(5)
    r2 = torch.randn(N)
    assert torch.allclose(r1, r2)


def test_save_rng_unseeded_does_not_change_global():
    N = 100
    rand.seed(5)
    r1 = torch.randn(N)

    rand.seed(5)
    with rand.SavedRNG():
        torch.randn(N)
    r2 = torch.randn(N)
    assert torch.allclose(r1, r2)


def test_save_rng_seeded_does_not_change_global():
    N = 100
    rand.seed(5)
    r1 = torch.randn(N)

    rand.seed(5)
    with rand.SavedRNG(0):
        torch.randn(N)
    r2 = torch.randn(N)
    assert torch.allclose(r1, r2)


def test_save_rng_seeded_local():
    mppi_rng = rand.SavedRNG(0)
    goal_rng = rand.SavedRNG(1)

    with goal_rng:
        goal_1_together = torch.randn(1)
        goal_2_together = torch.randn(1)

    with mppi_rng:
        mppi_1_together = torch.randn(1)
        mppi_2_together = torch.randn(1)

    mppi_rng = rand.SavedRNG(0)
    goal_rng = rand.SavedRNG(1)

    # should be same as
    with goal_rng:
        goal_1_interleaved = torch.randn(1)

    with mppi_rng:
        mppi_1_interleaved = torch.randn(1)

    with goal_rng:
        goal_2_interleaved = torch.randn(1)

    with mppi_rng:
        mppi_2_interleaved = torch.randn(1)

    assert goal_1_together == goal_1_interleaved
    assert goal_2_together == goal_2_interleaved
    assert mppi_1_together == mppi_1_interleaved
    assert mppi_2_together == mppi_2_interleaved


if __name__ == "__main__":
    test_seed()
    test_save_rng_seeded_does_not_change_global()
    test_save_rng_unseeded_does_not_change_global()
    test_save_rng_seeded_local()
