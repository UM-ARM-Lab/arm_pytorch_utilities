import torch
from arm_pytorch_utilities import grad




def test_jacobian():
    def f1(x):
        return torch.cat((x ** 2, (2 * x[:, 0] + torch.log(x[:, 1])).view(-1, 1)), dim=1)

    nx = 2
    nout_f1 = nx + 1
    x = torch.rand(nx)
    j1 = grad.jacobian(f1, x)
    j2 = grad.jacobian(f1, x.view(1, -1))
    assert j1.shape == (nout_f1, nx)
    assert torch.allclose(j1, j2)
    assert torch.allclose(j1[:, 0], torch.tensor([2 * x[0], 0, 2.]))
    assert torch.allclose(j1[:, 1], torch.tensor([0., 2 * x[1], 1. / x[1]]))

    nx = 5
    nout_f2 = 3
    net = torch.nn.Linear(nx, nout_f2, bias=False)

    def f2(x):
        return net(x)

    x = torch.randn(1, nx)
    # test that it doesn't matter if other operations are ran on the network before calculating jacobian
    f2(x)
    j3 = grad.jacobian(f2, x)
    assert j3.shape == (nout_f2, nx)
    assert torch.allclose(j3, net.weight)

    # can even have existing gradients calculated from some other backward branch
    y = f2(x)
    y.sum().backward()
    j3 = grad.jacobian(f2, x)
    assert torch.allclose(j3, net.weight)


def test_batch_jacobian():
    # test batch jacobian
    N = 10
    nx = 5
    nout_f2 = 3
    net = torch.nn.Linear(nx, nout_f2, bias=False)

    x = torch.rand((N, nx))
    jall = grad.batch_jacobian(net, x)
    assert jall.shape == (N, nout_f2, nx)
    for i in range(N):
        j = grad.jacobian(net, x[i])
        assert torch.allclose(jall[i], j)


if __name__ == "__main__":
    # test_jacobian()
    test_batch_jacobian()
