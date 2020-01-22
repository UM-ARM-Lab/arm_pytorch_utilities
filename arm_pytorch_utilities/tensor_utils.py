import torch


def ensure_tensor(device, dtype, *args):
    tensors = tuple(torch.tensor(x, device=device, dtype=dtype) for x in args)
    return tensors if len(tensors) > 1 else tensors[0]
