import numpy as np


class BaseSmoothOracle(object):
    """
    Base class for implementation of oracles.
    """

    def func(self, x):
        """
        Computes the value of function at point x.
        """
        raise NotImplementedError('Func oracle is not implemented.')

    def grad(self, x):
        """
        Computes the gradient at point x.
        """
        raise NotImplementedError('Grad oracle is not implemented.')

    def hess(self, x):
        """
        Computes the Hessian matrix at point x.
        """
        raise NotImplementedError('Hessian oracle is not implemented.')

    def func_directional(self, x, d, alpha):
        """
        Computes phi(alpha) = f(x + alpha*d).
        """
        return np.squeeze(self.func(x + alpha * d))

    def grad_directional(self, x, d, alpha):
        """
        Computes phi'(alpha) = (f(x + alpha*d))'_{alpha}
        """
        return np.squeeze(self.grad(x + alpha * d).dot(d))


class BarrierOracle(BaseSmoothOracle):

    def __init__(self, A: np.ndarray, b: np.ndarray, reg: float, t: float):
        self.A = A
        self.b = b
        self.reg = reg
        self.t = t

        self.ax = lambda x: np.dot(A, x)
        self.atx = lambda x: np.dot(A.T, x)

    def antider_func(self, pt: np.ndarray):
        x, u = np.split(pt, 2)
        diff = self.ax(x) - self.b
        return 0.5 * np.sum(diff ** 2) + self.reg * np.sum(u)

    def func(self, pt: np.ndarray):
        x, u = np.split(pt, 2)
        logp = np.log(u + x)
        logn = np.log(u - x)
        return self.t * self.antider_func(pt) - np.sum(logp + logn)

    def grad(self, pt: np.ndarray):
        x, u = np.split(pt, 2)
        invp = 1 / (u + x)
        invn = 1 / (u - x)
        gx = self.t * self.atx(self.ax(x) - self.b) - invp + invn
        gu = self.t * self.reg * np.ones_like(u)
        return np.concatenate((gx, gu))

    def hess(self, pt: np.ndarray):
        x, u = np.split(pt, 2)
        invp2 = 1 / (u + x) ** 2
        invn2 = 1 / (u - x) ** 2
        sum_inv = invp2 + invn2
        diff_inv = invp2 - invn2
        hu = np.diag(sum_inv)
        hx = self.t * (self.A.T @ self.A) + np.diag(sum_inv)
        hxu = np.diag(diff_inv)
        top = np.hstack((hx, hxu))
        bot = np.hstack((hxu, hu))
        return np.vstack((top, bot))


def lasso_duality_gap(x, Ax_b, ATAx_b, b, regcoef):
    """
    Estimates f(x) - f* via duality gap for 
        f(x) := 0.5 * ||Ax - b||_2^2 + regcoef * ||x||_1.
    """
    # TODO: implement.
    norm_inf = np.max(np.abs(ATAx_b))
    mu_scaling = regcoef / norm_inf if norm_inf > 1e-5 else 1.0
    mu = np.minimum(1.0, mu_scaling) * Ax_b

    term1 = 0.5 * np.sum(Ax_b ** 2)
    term2 = regcoef * np.sum(np.abs(x))
    term3 = 0.5 * np.sum(mu ** 2)
    term4 = np.dot(b, mu)

    gap = term1 + term2 + term3 + term4
    return gap
