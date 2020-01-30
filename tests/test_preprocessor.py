from arm_pytorch_utilities import preprocess
import torch


class Multiplier(preprocess.SingleTransformer):
    def __init__(self, scale):
        self.scale = scale

    def fit(self, X):
        pass

    def transform(self, X):
        return X * self.scale

    def inverse_transform(self, X):
        return X / self.scale


def verify_scaler_tsf(x, y, tsf, scale=1.):
    tsf.fit(x, y)
    nx = x.shape[1]
    ny = y.shape[1]
    xx, yy, _ = tsf.transform(x, y)
    v, _ = xx.min(0)
    assert torch.allclose(v, torch.zeros(nx))
    v, _ = xx.max(0)
    assert torch.allclose(v, torch.ones(nx) * scale)
    v, _ = yy.min(0)
    assert torch.allclose(v, torch.zeros(ny))
    v, _ = yy.max(0)
    assert torch.allclose(v, torch.ones(ny) * scale)
    yyy = tsf.invert_transform(yy, x)
    assert torch.allclose(y, yyy, atol=1e-7)


def test_preprocess_compose():
    N = 100
    nx = 3
    ny = 2
    scale = 5.5
    x = torch.randn((N, nx))
    y = torch.randn((N, ny))
    tsf1 = preprocess.PytorchTransformer(preprocess.MinMaxScaler())
    tsf2 = preprocess.PytorchTransformer(Multiplier(scale))

    combo1 = preprocess.Compose([tsf1, tsf2])
    # multiplier applied afterwards
    verify_scaler_tsf(x, y, combo1, scale)

    combo2 = preprocess.Compose([tsf2, tsf1])
    # multiplier applied before scaler
    verify_scaler_tsf(x, y, combo2)


def test_min_max_scaler():
    N = 100
    nx = 3
    ny = 2
    x = torch.randn((N, nx))
    y = torch.randn((N, ny))
    tsf = preprocess.PytorchTransformer(preprocess.MinMaxScaler())
    verify_scaler_tsf(x, y, tsf)


if __name__ == "__main__":
    test_min_max_scaler()
    test_preprocess_compose()
