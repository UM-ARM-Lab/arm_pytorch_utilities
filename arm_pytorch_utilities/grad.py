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
    ny = 1 if len(y.shape) < 2 else y.shape[1]
    x = x.repeat(ny, 1)
    x.requires_grad_(True)
    y = f(x)
    if ny == 1:
        y.backward()
    else:
        y.backward(torch.eye(ny, dtype=y.dtype, device=y.device))
    return x.grad.data


def batch_jacobian(f, x):
    """
    Compute jacobian for a batch of x
    :param f: f(x) -> y (supporting batch input B x <some num> x nx), such as a NN
    :param x: B x nx
    :return: df/dx(x) B x ny x nx jacobian of f evaluated at each x
    """
    y = f(x)
    ny = y.shape[1]

    x = x.unsqueeze(1)  # b, 1, nx
    b = x.shape[0]
    x = x.repeat(1, ny, 1)  # b, ny, nx
    x.requires_grad_(True)
    y = f(x)
    input_val = torch.eye(ny, dtype=y.dtype, device=y.device).repeat(b, 1, 1)
    y.backward(input_val)
    return x.grad.data
