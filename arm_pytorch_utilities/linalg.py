import torch
from torch.autograd import Function
import numpy as np
import scipy.linalg
import logging

logger = logging.getLogger(__name__)


def batch_quadratic_product(X, A):
    """
    Batch multiplication of x^T A x.
    :param X: N x nx where each x is a row in X
    :param A: nx x nx
    :returns N x 1 product of each x^T A x
    """
    return torch.einsum('ij,kj,ik->i', X, A, X)


def batch_outer_product(u, v):
    """
    Batch outer product uv^T
    :param u: N x nx
    :param v: N x nx
    :return: N x nx x nx where each 'row' is the outer product of each row of u and v
    """
    return torch.einsum('ij,ik->ijk', u, v)


def batch_batch_product(X, A):
    """
    Batch multiplication of y = Ax where both A and x are batched
    :param X: N x nx
    :param A: N x ny x nx
    :return: N x ny product of each A with each x
    """
    return torch.einsum('ijk,ik->ij', A, X)


def kronecker_product(t1, t2):
    """
    Computes the Kronecker product between two tensors.
    See https://en.wikipedia.org/wiki/Kronecker_product
    """
    t1_height, t1_width = t1.size()
    t2_height, t2_width = t2.size()
    out_height = t1_height * t2_height
    out_width = t1_width * t2_width

    tiled_t2 = t2.repeat(t1_height, t1_width)
    expanded_t1 = (
        t1.unsqueeze(2)
            .unsqueeze(3)
            .repeat(1, t2_height, t2_width, 1)
            .view(out_height, out_width)
    )

    return expanded_t1 * tiled_t2


def cov(x, rowvar=False, bias=False, ddof=None, aweights=None):
    """Estimates covariance matrix like numpy.cov"""
    # ensure at least 2D
    if x.dim() == 1:
        x = x.view(-1, 1)

    # treat each column as a data point, each row as a variable
    if rowvar and x.shape[0] != 1:
        x = x.t()

    if ddof is None:
        if bias == 0:
            ddof = 1
        else:
            ddof = 0

    w = aweights
    if w is not None:
        if not torch.is_tensor(w):
            w = torch.tensor(w, dtype=torch.float)
        w_sum = torch.sum(w)
        avg = torch.sum(x * (w / w_sum)[:, None], 0)
    else:
        avg = torch.mean(x, 0)

    # Determine the normalization
    if w is None:
        fact = x.shape[0] - ddof
    elif ddof == 0:
        fact = w_sum
    elif aweights is None:
        fact = w_sum - ddof
    else:
        fact = w_sum - ddof * torch.sum(w * w) / w_sum

    xm = x.sub(avg.expand_as(x))

    if w is None:
        X_T = xm.t()
    else:
        X_T = torch.mm(torch.diag(w), xm).t()

    c = torch.mm(X_T, xm)
    c = c / fact

    return c.squeeze()


class MatrixSquareRoot(Function):
    """Square root of a positive definite matrix.
    From https://github.com/steveli/pytorch-sqrtm
    NOTE: matrix square root is not differentiable for matrices with
          zero eigenvalues.
    """

    @staticmethod
    def forward(ctx, input):
        m = input.detach().numpy().astype(np.float_)
        sqrtm = torch.from_numpy(scipy.linalg.sqrtm(m).real).type_as(input)
        ctx.save_for_backward(sqrtm)
        return sqrtm

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = None
        if ctx.needs_input_grad[0]:
            sqrtm, = ctx.saved_tensors
            sqrtm = sqrtm.data.numpy().astype(np.float_)
            gm = grad_output.data.numpy().astype(np.float_)

            # Given a positive semi-definite matrix X,
            # since X = X^{1/2}X^{1/2}, we can compute the gradient of the
            # matrix square root dX^{1/2} by solving the Sylvester equation:
            # dX = (d(X^{1/2})X^{1/2} + X^{1/2}(dX^{1/2}).
            grad_sqrtm = scipy.linalg.solve_sylvester(sqrtm, sqrtm, gm)

            grad_input = torch.from_numpy(grad_sqrtm).type_as(grad_output.data)
        return grad_input


sqrtm = MatrixSquareRoot.apply


class GELS(Function):
    """ Efficient implementation of gels from
        Nanxin Chen
        bobchennan@gmail.com
    """

    @staticmethod
    def forward(ctx, A, b):
        # A: (..., M, N)
        # b: (..., M, K)
        # https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/ops/linalg_ops.py#L267
        u = torch.cholesky(torch.matmul(A.transpose(-1, -2), A), upper=True)
        ret = torch.cholesky_solve(torch.matmul(A.transpose(-1, -2), b), u, upper=True)
        ctx.save_for_backward(u, ret, A, b)
        return ret

    @staticmethod
    def backward(ctx, grad_output):
        # https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/ops/linalg_grad.py#L223
        chol, x, a, b = ctx.saved_tensors
        z = torch.cholesky_solve(grad_output, chol, upper=True)
        xzt = torch.matmul(x, z.transpose(-1, -2))
        zx_sym = xzt + xzt.transpose(-1, -2)
        grad_A = - torch.matmul(a, zx_sym) + torch.matmul(b, z.transpose(-1, -2))
        grad_b = torch.matmul(a, z)
        return grad_A, grad_b


def _apply_weights(X, Y, weights):
    if weights is not None:
        # assume errors are uncorrelated so weight is given as a vector rather than matrix
        w = torch.sqrt(weights).view(-1, 1)
        # multiplying a diagonal matrix is equal to multiplying each row by the corresponding weight
        X = w * X
        Y = w * Y
    return X, Y


def ls(X, Y, weights=None):
    X, Y = _apply_weights(X, Y, weights)

    # currently no gradient support for gels
    # params, _ = torch.gels(Y, X)
    params = GELS.apply(X, Y)
    # params = torch.solve(Y, X)
    # params = torch.mm(X.pinverse(), Y)

    return params


SYMMETRIC_CORRECTION_NORM_THRESHOLD = 1e-3


def ls_cov(X, Y, weights=None, make_symmetric=True, sigreg=1e-4):
    X, Y = _apply_weights(X, Y, weights)

    pinvXX = X.pinverse()
    params = (pinvXX @ Y).t()

    # estimate covariance according to: http://users.stat.umn.edu/~helwig/notes/mvlr-Notes.pdf (see up to slide 66)
    # hat/projection matrix - Yhat = H*Y
    H = X @ pinvXX

    N = X.shape[0]
    n = X.shape[1]
    # degrees of freedom
    dof = N - n
    assert dof >= 0
    # estimated error covariance (unbiased estimate of error cov matrix Sigma
    # regularize for ill-conditioned matrices
    a = torch.eye(N, dtype=H.dtype, device=X.device) - H
    error_covariance = Y.t() @ a @ Y / dof
    error_covariance += torch.eye(error_covariance.shape[0], dtype=H.dtype, device=X.device) * sigreg

    XXXX = X.t() @ X
    # regularize
    XXXX += torch.eye(XXXX.shape[0], dtype=H.dtype, device=X.device) * sigreg

    if make_symmetric:
        # correct to be symmetric (if needed)
        error_covariance_sym = (error_covariance + error_covariance.t()) / 2
        XXXX_sym = (XXXX + XXXX.t()) / 2

        e_correction_error = (error_covariance_sym - error_covariance).norm()
        x_correction_error = (XXXX_sym - XXXX).norm()
        if e_correction_error > SYMMETRIC_CORRECTION_NORM_THRESHOLD or x_correction_error > SYMMETRIC_CORRECTION_NORM_THRESHOLD:
            logger.debug('unsymmetric covariances with error %f and %f', e_correction_error, x_correction_error)
        XXXX = XXXX_sym
        error_covariance = error_covariance_sym

    # TODO might be able to use cholesky decomp here since XXXX > 0
    covariance = kronecker_product(error_covariance, XXXX.inverse())

    return params, covariance


if __name__ == "__main__":
    d = torch.tensor([[1, 2], [3, 4]], dtype=torch.float)
    e = torch.eye(3, dtype=torch.float)
    f = kronecker_product(d.t(), e)
    print(f)
