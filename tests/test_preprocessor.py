from arm_pytorch_utilities import preprocess
import torch


def test_min_max_scaler():
    N = 100
    nx = 3
    ny = 2
    x = torch.randn((N, nx))
    y = torch.randn((N, ny))
    tsf = preprocess.PytorchTransformer(preprocess.MinMaxScaler())
    tsf.fit(x, y)
    xx, yy, _ = tsf.transform(x, y)

    v, _ = xx.min(0)
    assert torch.allclose(v, torch.zeros(nx))
    v, _ = xx.max(0)
    assert torch.allclose(v, torch.ones(nx))
    v, _ = yy.min(0)
    assert torch.allclose(v, torch.zeros(ny))
    v, _ = yy.max(0)
    assert torch.allclose(v, torch.ones(ny))

    yyy = tsf.invert_transform(yy, x)
    assert torch.allclose(y, yyy, atol=1e-7)


if __name__ == "__main__":
    test_min_max_scaler()
