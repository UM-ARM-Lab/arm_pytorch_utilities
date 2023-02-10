import torch
import numpy as np
from arm_pytorch_utilities.make_data import select
import scipy.io as sio
import os


def gt_model(xu, params, M, s, selector=None):
    if selector is None:
        selector = select.identity_select
    m = selector(xu)
    y = xu.new_empty((xu.shape[0], M))
    for i in range(s):
        in_cluster = m == i
        y[in_cluster] = torch.mm(xu[in_cluster], params[i].view(-1, M))
    return y, m


def sort(U, Y, labels):
    I = torch.argsort(labels, dim=0)
    U = U[I]
    Y = Y[I]
    labels = labels[I]

    return U, Y, labels


def make_input(p, N, variance):
    U = np.random.rand(N, p) * 8 - 4
    U = np.concatenate((U, np.ones((N, 1))), 1)

    E = np.random.randn(N, 1) * np.sqrt(variance)

    return U, E


def make(params, N=200, variance=0.05, sorted=False, selector=None, filename=None):
    M = params.shape[2]
    s = params.shape[0]
    p = params.shape[1] - 1

    U, E = make_input(p, N, variance)
    E = torch.tensor(E, dtype=params.dtype, device=params.device)
    U = torch.tensor(U, dtype=params.dtype, device=params.device)

    Y, labels = gt_model(U, params, M, s, selector)
    Y += E

    if sorted:
        U, Y, labels = sort(U, Y, labels)

    if filename is not None:
        joined_data = {'XU': U.numpy(), 'Y': Y.numpy(), 'labels': labels.numpy()}
        try:
            sio.savemat(filename, joined_data)
        except FileNotFoundError:
            os.makedirs(os.path.dirname(filename))
            sio.savemat(filename, joined_data)

    return U, Y, labels


def load(filename, max_N=None, to_load=('XU', 'Y', 'labels')):
    prev_data = sio.loadmat(filename)

    loaded = [prev_data[l][:max_N] for l in to_load]
    tensors = [torch.tensor(ll) for ll in loaded]

    return tensors
