"""Trajectory optimization (just LQR from GPS's source code)"""
import numpy as np
import scipy as sp
from numpy.linalg import LinAlgError
import logging

from arm_pytorch_utilities.policy.lin_gauss import LinearGaussianPolicy

LOGGER = logging.getLogger(__name__)

CHECK_FINITE = False  # False is faster


def lu_solve(L, U, A):
    """Solves LUX=A for X"""
    return sp.linalg.solve_triangular(U, sp.linalg.solve_triangular(L, A, lower=True,
                                                                    check_finite=CHECK_FINITE),
                                      check_finite=CHECK_FINITE)


def solve_psd(A, B, reg=0):
    """Solve AX=B via cholesky decomposition (A must be positive semidefinite)"""
    chol = sp.linalg.cholesky(A + reg * np.eye(A.shape[0]))
    return lu_solve(chol.T, chol, B)


def invert_psd(A, reg=0):
    """Invert a PSD matrix via Cholesky + Triangular solves"""
    return solve_psd(A, np.eye(A.shape[0]), reg=reg)


def lqr(cost, lgpolicy, dynamics,
        horizon, T, x, prevx, prevu,
        reg_mu, reg_del, del0, min_mu, discount,
        jacobian=None,
        max_time_varying_horizon=20):
    """
    TODO: Clean up args...

    Plain LQR.
    Returns:
        LinearGaussianPolicy: A new time-varying policy valid for <horizon> timesteps
        reg_mu: Value function regularization (from Tassa synthesis paper, currently unused)
        reg_del:
    """
    dX = prevx.shape[0]
    dU = prevu.shape[0]
    ix = slice(dX)
    iu = slice(dX, dX + dU)
    it = slice(dX + dU)
    ip = slice(dX + dU, dX + dU + dX)

    # Compute forward pass
    cv, Cm, Fd, fc, _, _ = \
        estimate_cost(T, cost, lgpolicy, dynamics, horizon, x, prevx, prevu,
                      max_time_varying_horizon, jacobian=jacobian)

    # Compute optimal policy with short horizon MPC.
    fail = True
    decrease_mu = True
    K = np.zeros((horizon, dU, dX))
    pSig = np.zeros((horizon, dU, dU))
    invPSig = np.zeros((horizon, dU, dU))
    cholPSig = np.zeros((horizon, dU, dU))
    k = np.zeros((horizon, dU))
    while fail:
        Vxx = np.zeros((dX, dX))
        Vx = np.zeros(dX)
        fail = False
        for t in range(horizon - 1, -1, -1):
            F = Fd[t]
            f = fc[t]

            Qtt = Cm[t]
            Qt = cv[t]

            Vxx = Vxx  # + reg_mu*np.eye(dX)  #Note: reg_mu is currently unused
            Qtt = Qtt + F.T.dot(Vxx.dot(F))
            Qt = Qt + F.T.dot(Vx) + F.T.dot(Vxx).dot(f)

            Qtt = 0.5 * (Qtt + Qtt.T)

            try:
                U = sp.linalg.cholesky(Qtt[iu, iu], check_finite=False)
                L = U.T
            except LinAlgError:
                fail = True
                decrease_mu = False
                break
            K[t] = -lu_solve(L, U, Qtt[iu, ix])
            k[t] = -lu_solve(L, U, Qt[iu])
            pSig[t] = lu_solve(L, U, np.eye(dU))
            invPSig[t] = invert_psd(pSig[t])
            cholPSig[t] = sp.linalg.cholesky(pSig[t], check_finite=False)

            # Compute value function.
            Vxx = discount * (Qtt[ix, ix] + Qtt[ix, iu].dot(K[t]))
            Vx = discount * (Qt[ix] + Qtt[ix, iu].dot(k[t]))
            Vxx = 0.5 * (Vxx + Vxx.T)

        # Tassa regularization scheme
        if fail:
            if reg_mu > 1e5:
                raise ValueError("Failed to find SPD solution")
            else:
                reg_del = max(del0, reg_del * del0)
                reg_mu = max(min_mu, reg_mu * reg_del)
                # LOGGER.debug('[LQR reg] Increasing mu -> %f', reg_mu)
        elif decrease_mu:
            reg_del = min(1 / del0, reg_del / del0)
            delmu = reg_del * reg_mu
            if delmu > min_mu:
                reg_mu = delmu
            else:
                reg_mu = min_mu
                # LOGGER.debug('[LQR reg] Decreasing mu -> %f', reg_mu)

    policy = LinearGaussianPolicy(K, k, pSig, cholPSig, invPSig)

    # plot new
    # self.forward(horizon, x, lgpolicy, T, hist_key='new')

    return policy, reg_mu, reg_del


def estimate_cost(cur_timestep, cost, lgpolicy, dynamics, horizon, x0, prevx, prevu, max_time_varying_horizon,
                  jacobian=None, time_varying_dynamics=True):
    """
    Returns cost derivatives and computed dynamics via a forward pass
    """
    # Cost + dynamics estimation

    H = horizon

    N = 1

    cholPSig = lgpolicy.chol_pol_covar
    # PSig = lgpolicy.pol_covar
    K = lgpolicy.K
    k = lgpolicy.k

    dX = K.shape[2]
    dU = K.shape[1]
    dT = dX + dU
    ix = slice(dX)
    iu = slice(dX, dX + dU)

    # Run forward pass

    # Allocate space.
    trajsig = np.zeros((H, dT, dT))
    mu = np.zeros((H, dT))

    trajsig[0, ix, ix] = np.zeros((dX, dX))
    mu[0, ix] = x0

    F = np.zeros((H, dX, dT))
    f = np.zeros((H, dX))
    dynsig = np.zeros((H, dX, dX))

    # HACK
    mu[-1, ix] = prevx
    mu[-1, iu] = prevu
    # Perform forward pass.
    for t in range(H):
        PSig = cholPSig[t].T.dot(cholPSig[t])
        trajsig[t] = np.r_[
            np.c_[trajsig[t, ix, ix], trajsig[t, ix, ix].dot(K[t].T)],
            np.c_[K[t].dot(trajsig[t, ix, ix]), K[t].dot(trajsig[t, ix, ix]).dot(K[t].T) + PSig]
        ]
        cur_action = K[t].dot(mu[t, ix]) + k[t]
        mu[t] = np.r_[mu[t, ix], cur_action]
        # mu[t] = np.r_[mu[t,ix], np.zeros(dU)]

        # Reuse old dynamics
        if not time_varying_dynamics:
            if t == 0:
                F[0], f[0], dynsig[0] = dynamics.get_dynamics(cur_timestep, prevx, prevu, x0, cur_action)
            F[t] = F[0]
            f[t] = f[0]
            dynsig[t] = dynsig[0]

        if t < H:
            # Estimate new dynamics here based on mu
            if time_varying_dynamics and t < max_time_varying_horizon:
                F[t], f[t], dynsig[t] = dynamics.get_dynamics(cur_timestep + t, mu[t - 1, ix], mu[t - 1, iu],
                                                              mu[t, ix], cur_action)
        if t < H - 1:
            trajsig[t + 1, ix, ix] = F[t].dot(trajsig[t]).dot(F[t].T) + dynsig[t]
            mu[t + 1, ix] = F[t].dot(mu[t]) + f[t]

    # cc = np.zeros((N, H))
    cv = np.zeros((N, H, dT))
    Cm = np.zeros((N, H, dT, dT))

    for n in range(N):
        # Get costs.
        newmu = mu[:, iu]
        l, lx, lu, lxx, luu, lux = cost.eval(mu[:, ix], newmu, cur_timestep, jac=jacobian)

        cv[n] = np.c_[lx, lu]
        Cm[n] = np.concatenate((np.c_[lxx, np.transpose(lux, [0, 2, 1])], np.c_[lux, luu]), axis=1)

        # Adjust for expanding cost around a sample.
        yhat = mu  # np.c_[mu[:,ix], newmu_u]
        rdiff = -yhat  # T x (X+U)
        rdiff_expand = np.expand_dims(rdiff, axis=2)  # T x (X+U) x 1
        cv_update = np.sum(Cm[n] * rdiff_expand, axis=1)  # T x (X+U)
        cv[n] += cv_update

    cv = np.mean(cv, axis=0)
    Cm = np.mean(Cm, axis=0)
    return cv, Cm, F, f, mu, trajsig


def dlqr(A, B, Q, R):
    """Solve the discrete time lqr controller.

    x[k+1] = A x[k] + B u[k]

    cost = sum x[k].T*Q*x[k] + u[k].T*R*u[k]
    """

    # ref Bertsekas, p.151

    # first, try to solve the ricatti equation
    X = sp.linalg.solve_discrete_are(A, B, Q, R)

    # compute the LQR gain
    K = sp.linalg.inv(B.T @ X @ B + R) @ (B.T @ X @ A)

    eigVals, eigVecs = sp.linalg.eig(A - B @ K)

    return K, X, eigVals
