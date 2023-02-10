""" This file defines the linear Gaussian policy class from GPS. """
import numpy as np
import scipy as sp
from arm_pytorch_utilities.array_utils import check_shape


class LinearGaussianPolicy:
    """
    Time-varying linear Gaussian policy.
    U = K*x + k + noise, where noise ~ N(0, chol_pol_covar)
    """

    def __init__(self, K, k, pol_covar, chol_pol_covar, inv_pol_covar, cache_kldiv_info=False):
        # Assume K has the correct shape, and make sure others match.
        self.T = K.shape[0]
        self.dU = K.shape[1]
        self.dX = K.shape[2]

        check_shape(k, (self.T, self.dU))
        check_shape(pol_covar, (self.T, self.dU, self.dU))
        check_shape(chol_pol_covar, (self.T, self.dU, self.dU))
        check_shape(inv_pol_covar, (self.T, self.dU, self.dU))

        self.K = K
        self.k = k
        self.pol_covar = pol_covar
        self.chol_pol_covar = chol_pol_covar
        self.inv_pol_covar = inv_pol_covar

        # KL Div stuff
        if cache_kldiv_info:
            self.logdet_psig = np.zeros(self.T)
            self.precision = np.zeros((self.T, self.dU, self.dU))
            for t in range(self.T):
                self.logdet_psig[t] = 2 * sum(np.log(np.diag(chol_pol_covar[t])))
                self.precision[t] = sp.linalg.solve_triangular(chol_pol_covar[t],
                                                               sp.linalg.solve_triangular(chol_pol_covar[t].T,
                                                                                          np.eye(self.dU), lower=True,
                                                                                          check_finite=False),
                                                               check_finite=False)

    def act(self, x, obs, t, noise=None, sample=None):
        """
        Return an action for a state.
        Args:
            x: State vector.
            obs: Observation vector.
            t: Time step.
            noise: Action noise. This will be scaled by the variance.
        """
        # import pdb; pdb.set_trace()
        u = self.K[t].dot(x) + self.k[t]
        u += self.chol_pol_covar[t].T.dot(noise)
        return u

    def fold_k(self, noise):
        """
        Fold noise into k.
        Args:
            noise: A T x Du noise vector with mean 0 and variance 1.
        Returns:
            k: A T x dU bias vector.
        """
        k = np.zeros_like(self.k)
        for i in range(self.T):
            scaled_noise = self.chol_pol_covar[i].T.dot(noise[i])
            k[i] = scaled_noise + self.k[i]
        return k

    def nans_like(self):
        """
        Returns:
            A new linear Gaussian policy object with the same dimensions
            but all values filled with NaNs.
        """
        policy = LinearGaussianPolicy(
            np.zeros_like(self.K), np.zeros_like(self.k),
            np.zeros_like(self.pol_covar), np.zeros_like(self.chol_pol_covar),
            np.zeros_like(self.inv_pol_covar)
        )
        policy.K.fill(np.nan)
        policy.k.fill(np.nan)
        policy.pol_covar.fill(np.nan)
        policy.chol_pol_covar.fill(np.nan)
        policy.inv_pol_covar.fill(np.nan)
        return policy
