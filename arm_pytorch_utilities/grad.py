import torch


def jacobian(f, x):
    """
    Compute the jacobian of f evaluated at x. If f is a network or has parameters, freezing it
    (set requires_grad = False) improves performance (especially on large networks)

    From https://gist.github.com/sbarratt/37356c46ad1350d4c30aefbd488a4faa
    :param f: f(x) -> y
    :param x: 1 x nx or nx input to evaluate the jacobian at
    :return: df/dx(x) ny x nx jacobian of f evaluated at x
    """
    if x.dim() < 2:
        x = x.view(1, -1)
    y = f(x)
    noutputs = y.shape[1]
    x = x.repeat(noutputs, 1)
    x.requires_grad_(True)
    y = f(x)
    y.backward(torch.eye(noutputs, dtype=y.dtype))
    return x.grad.data
