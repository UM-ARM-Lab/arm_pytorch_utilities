import torch

def pairwise_distance(x):
    mat_square = torch.mm(x, x.t())
    diag = torch.diagonal(mat_square)
    diag = diag.expand_as(mat_square)
    # element i,j is L2 norm from ith row to jth row in mat
    dist_mat = (diag + diag.t() - 2 * mat_square)
    return dist_mat