import torch
import functools
import numpy as np


def is_tensor_like(x):
    return torch.is_tensor(x) or type(x) is np.ndarray


def ensure_2d_input(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        args = [v.reshape(1, -1) if (is_tensor_like(v) and len(v.shape) == 1) else v for v in args]
        return func(*args, **kwargs)

    return wrapper


def handle_batch_input(func):
    """For func that expect 2D input, handle input that have more than 2 dimensions by flattening them temporarily"""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # assume inputs that are tensor-like have compatible shapes and is represented by the first argument
        batch_dims = []
        for arg in args:
            if is_tensor_like(arg) and len(arg.shape) > 2:
                batch_dims = arg.shape[:-1]  # last dimension is type dependent; all previous ones are batches
                break
        # no batches; just return normally
        if not batch_dims:
            return func(*args, **kwargs)

        # reduce all batch dimensions down to the first one
        args = [v.view(-1, v.shape[-1]) if (is_tensor_like(v) and len(v.shape) > 2) else v for v in args]
        ret = func(*args, **kwargs)
        # restore original batch dimensions; keep variable dimension (nx)
        if type(ret) is tuple:
            ret = [v if (not is_tensor_like(v) or len(v.shape) == 0) else (
                v.view(*batch_dims, v.shape[-1]) if len(v.shape) == 2 else v.view(*batch_dims)) for v in ret]
        else:
            if is_tensor_like(ret):
                if len(ret.shape) == 2:
                    ret = ret.view(*batch_dims, ret.shape[-1])
                else:
                    ret = ret.view(*batch_dims)
        return ret

    return wrapper


def ensure_tensor(device, dtype, *args):
    tensors = tuple(
        x.to(device=device, dtype=dtype) if torch.is_tensor(x) else
        torch.tensor(x, device=device, dtype=dtype)
        for x in args)
    return tensors if len(tensors) > 1 else tensors[0]


def ensure_diagonal(Q, dim):
    """
    Ensure that Q is a (dim x dim) diagonal matrix either with values of Q, or with eye * Q if Q is scalar
    :param Q: (dim x dim) tensor or numpy array, or a (dim) vector of the diagonal, or a scalar to scale an identity
    matrix by
    :param dim: size of matrix
    :return: (dim x dim) diagonal matrix
    """
    if not torch.is_tensor(Q):
        if type(Q) in (int, float):
            Q = torch.eye(dim) * Q
        else:
            Q = torch.tensor(Q)
    if len(Q.shape) == 1:
        if Q.shape[0] is not dim:
            raise RuntimeError("Expect {} sized diagonal vector but given {}".format(dim, Q.shape[0]))
        Q = torch.diag(Q)
    return Q
