import torch


def ensure_tensor(device, dtype, *args):
    tensors = tuple(torch.tensor(x, device=device, dtype=dtype) for x in args)
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
    if len(Q.shape) is 1:
        if Q.shape[0] is not dim:
            raise RuntimeError("Expect {} sized diagonal vector but given {}".format(dim, Q.shape[0]))
        Q = torch.diag(Q)
    return Q
