from collections import defaultdict
from time import time
import numpy as np
from numpy.linalg import LinAlgError
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize.linesearch import scalar_search_wolfe2
from oracles import BarrierOracle
from oracles import lasso_duality_gap


class LineSearchTool(object):
    """
    Line search tool for adaptively tuning the step size of the algorithm.

    method : String containing 'Wolfe', 'Armijo' or 'Constant'
        Method of tuning step-size.
        Must be be one of the following strings:
            - 'Wolfe' -- enforce strong Wolfe conditions;
            - 'Armijo" -- adaptive Armijo rule;
            - 'Constant' -- constant step size.
    kwargs :
        Additional parameters of line_search method:

        If method == 'Wolfe':
            c1, c2 : Constants for strong Wolfe conditions
            alpha_0 : Starting point for the backtracking procedure
                to be used in Armijo method in case of failure of Wolfe method.
        If method == 'Armijo':
            c1 : Constant for Armijo rule
            alpha_0 : Starting point for the backtracking procedure.
        If method == 'Constant':
            c : The step size which is returned on every step.
    """
    def __init__(self, method='Armijo', **kwargs):
        self._method = method
        if self._method == 'Wolfe':
            self.c1 = kwargs.get('c1', 1e-4)
            self.c2 = kwargs.get('c2', 0.9)
            self.alpha_0 = kwargs.get('alpha_0', 1.0)
        elif self._method == 'Armijo':
            self.c1 = kwargs.get('c1', 1e-4)
            self.alpha_0 = kwargs.get('alpha_0', 1.0)
        elif self._method == 'Constant':
            self.c = kwargs.get('c', 1.0)
        else:
            raise ValueError('Unknown method {}'.format(method))

    @classmethod
    def from_dict(cls, options):
        if type(options) != dict:
            raise TypeError('LineSearchTool initializer must be of type dict')
        return cls(**options)

    def to_dict(self):
        return self.__dict__

    def line_search(self, oracle, x_k, d_k, previous_alpha=None):
        """
        Finds the step size alpha for a given starting point x_k
        and for a given search direction d_k that satisfies necessary
        conditions for phi(alpha) = oracle.func(x_k + alpha * d_k).

        Parameters
        ----------
        oracle : BaseSmoothOracle-descendant object
            Oracle with .func_directional() and .grad_directional() methods implemented for computing
            function values and its directional derivatives.
        x_k : np.array
            Starting point
        d_k : np.array
            Search direction
        previous_alpha : float or None
            Starting point to use instead of self.alpha_0 to keep the progress from
             previous steps. If None, self.alpha_0, is used as a starting point.

        Returns
        -------
        alpha : float or None if failure
            Chosen step size
        """
        derphi = lambda alpha: oracle.grad_directional(x_k, d_k, alpha)
        phi = lambda alpha: oracle.func_directional(x_k, d_k, alpha)
        phi0, derphi0 = phi(0), derphi(0)

        if self._method == 'Constant':
            alpha = self.c
        elif self._method == 'Armijo':
            alpha = self.alpha_0 if previous_alpha is None else previous_alpha
            derphi0 = derphi(0)
            while phi(alpha) > phi(0) + self.c1 * alpha * derphi0:
                alpha = alpha / 2
        elif self._method == 'Wolfe':
            alpha = scalar_search_wolfe2(phi, derphi, phi0, None, derphi0, c1=self.c1, c2=self.c2)
            alpha = alpha[0]
            if alpha is None:
                alpha = self.alpha_0 if previous_alpha is None else previous_alpha
                derphi0 = np.dot(oracle.grad(x_k), d_k)
                while phi(alpha) > phi(0) + self.c1 * alpha * derphi0:
                    alpha = alpha / 2
        return alpha


def get_line_search_tool(line_search_options=None):
    if line_search_options:
        if type(line_search_options) is LineSearchTool:
            return line_search_options
        else:
            return LineSearchTool.from_dict(line_search_options)
    else:
        return LineSearchTool()




def barrier_method_lasso(A, b, reg_coef, x_0, u_0, tolerance=1e-5,
                         tolerance_inner=1e-8, max_iter=100,
                         max_iter_inner=20, t_0=1, gamma=10,
                         c1=1e-4, lassodualitygap=None,
                         trace=False, display=False):
    """
    Log-barrier method for solving the problem:
        minimize    f(x, u) := 1/2 * ||Ax - b||_2^2 + reg_coef * \sum_i u_i
        subject to  -u_i <= x_i <= u_i.

    The method constructs the following barrier-approximation of the problem:
        phi_t(x, u) := t * f(x, u) - sum_i( log(u_i + x_i) + log(u_i - x_i) )
    and minimize it as unconstrained problem by Newton's method.

    In the outer loop `t` is increased and we have a sequence of approximations
        { phi_t(x, u) } and solutions { (x_t, u_t)^{*} } which converges in `t`
    to the solution of the original problem.

    Parameters
    ----------
    A : np.array
        Feature matrix for the regression problem.
    b : np.array
        Given vector of responses.
    reg_coef : float
        Regularization coefficient.
    x_0 : np.array
        Starting value for x in optimization algorithm.
    u_0 : np.array
        Starting value for u in optimization algorithm.
    tolerance : float
        Epsilon value for the outer loop stopping criterion:
        Stop the outer loop (which iterates over `k`) when
            `duality_gap(x_k) <= tolerance`
    tolerance_inner : float
        Epsilon value for the inner loop stopping criterion.
        Stop the inner loop (which iterates over `l`) when
            `|| \nabla phi_t(x_k^l) ||_2^2 <= tolerance_inner * \| \nabla \phi_t(x_k) \|_2^2 `
    max_iter : int
        Maximum number of iterations for interior point method.
    max_iter_inner : int
        Maximum number of iterations for inner Newton's method.
    t_0 : float
        Starting value for `t`.
    gamma : float
        Multiplier for changing `t` during the iterations:
        t_{k + 1} = gamma * t_k.
    c1 : float
        Armijo's constant for line search in Newton's method.
    lasso_duality_gap : callable object or None.
        If callable the signature is lasso_duality_gap(x, Ax_b, ATAx_b, b, regcoef)
        Returns duality gap value for estimating the progress of method.
    trace : bool
        If True, the progress information is appended into history dictionary
        during training. Otherwise None is returned instead of history.
    display : bool
        If True, debug information is displayed during optimization.
        Printing format is up to a student and is not checked in any way.

    Returns
    -------
    (x_star, u_star) : tuple of np.array
        The point found by the optimization procedure.
    message : string
        "success" or the description of error:
            - 'iterations_exceeded': if after max_iter iterations of the method x_k still doesn't satisfy
                the stopping criterion.
            - 'computational_error': in case of getting Infinity or None value during the computations.
    history : dictionary of lists or None
        Dictionary containing the progress information or None if trace=False.
        Dictionary has to be organized as follows:
            - history['time'] : list of floats, containing time in seconds passed from the start of the method
            - history['func'] : list of function values f(x_k) on every **outer** iteration of the algorithm
            - history['duality_gap'] : list of duality gaps
            - history['x'] : list of np.arrays, containing the trajectory of the algorithm. ONLY STORE IF x.size <= 2
    """
    history = defaultdict(list) if trace else None
    x = np.copy(x_0)
    u = np.copy(u_0)
    t = t_0

    if display:
        print('Optimization debug information')
    start_time = time()

    ldg_func = lassodualitygap if lassodualitygap is not None else lasso_duality_gap
    Ax_b = lambda y: A @ y - b
    ATAx_b = lambda y: A.T @ (A @ y - b)
    calc_ldg = lambda a: ldg_func(a, Ax_b(a), ATAx_b(a), b, reg_coef)
    line_search_opts = {'method': 'Armijo', 'c1': c1}
    oracle = BarrierOracle(A, b, reg_coef, t_0)
    ldg_val = calc_ldg(x)
    elapsed_time = time() - start_time

    if trace:
        history['func'].append([oracle.antider_func(np.concatenate([x, u]))])
        history['time'].append([elapsed_time])
        history['duality_gap'].append([ldg_val])
        if x.size <= 2:
            history['x'].append([x])

    for _ in range(max_iter):
        if ldg_val < tolerance:
            return (x, u), 'success', history

        oracle.t = t
        x_new, msg_new, _ = newton(oracle, np.concatenate([x, u]),
                                   tolerance_inner, max_iter_inner,
                                   line_search_opts)
        x, u = np.array_split(x_new, 2)
        if msg_new == 'computational_error':
            return (x, u), 'computational_error', history

        t *= gamma
        ldg_val = calc_ldg(x)
        elapsed_time = time() - start_time
        if trace:
            history['func'].append([oracle.antider_func(np.concatenate([x, u]))])
            history['time'].append([elapsed_time])
            history['duality_gap'].append([ldg_val])
            if x.size <= 2:
                history['x'].append([x])

    if ldg_val < tolerance:
        return (x, u), 'success', history

    return (x, u), 'iterations_exceeded', history






def optimal_alpha(v: np.ndarray, g: np.ndarray):
    """
    Calculate the optimal alpha for the given vectors and gradients.
    """
    n = len(v) // 2
    x = v[:n]
    u = v[n:]
    gx = g[:n]
    gu = g[n:]

    theta = 0.99
    alphas = np.array([1.])
    neg_mask, pos_mask = gx < -gu, gx > gu

    pos_alpha = theta * (u[pos_mask] - x[pos_mask]) / (gx[pos_mask] - gu[pos_mask])
    neg_alpha = theta * (x[neg_mask] + u[neg_mask]) / (-gx[neg_mask] - gu[neg_mask])
    alphas = np.concatenate((alphas, pos_alpha, neg_alpha))

    return np.min(alphas)


def newton(oracle, x_0, tolerance=1e-5, max_iter=100, line_search_options=None, trace=False, display=False):
    """
    Newton's optimization method.

    Parameters
    ----------
    oracle : BaseSmoothOracle-descendant object
        Oracle with .func(), .grad() and .hess() methods implemented for computing
        function value, its gradient and Hessian respectively. If the Hessian
        returned by the oracle is not positive-definite method stops with message="newton_direction_error"
    x_0 : np.array
        Starting point for optimization algorithm
    tolerance : float
        Epsilon value for stopping criterion.
    max_iter : int
        Maximum number of iterations.
    line_search_options : dict, LineSearchTool or None
        Dictionary with line search options. See LineSearchTool class for details.
    trace : bool
        If True, the progress information is appended into history dictionary during training.
        Otherwise None is returned instead of history.
    display : bool
        If True, debug information is displayed during optimization.

    Returns
    -------
    x_star : np.array
        The point found by the optimization procedure
    message : string
        'success' or the description of error:
            - 'iterations_exceeded': if after max_iter iterations of the method x_k still doesn't satisfy
                the stopping criterion.
            - 'newton_direction_error': in case of failure of solving linear system with Hessian matrix (e.g. non-invertible matrix).
            - 'computational_error': in case of getting Infinity or None value during the computations.
    history : dictionary of lists or None
        Dictionary containing the progress information or None if trace=False.
        Dictionary has to be organized as follows:
            - history['time'] : list of floats, containing time passed from the start of the method
            - history['func'] : list of function values f(x_k) on every step of the algorithm
            - history['grad_norm'] : list of values Euclidian norms ||g(x_k)|| of the gradient on every step of the algorithm
            - history['x'] : list of np.arrays, containing the trajectory of the algorithm. ONLY STORE IF x.size <= 2

    Example:
    --------
    >> oracle = QuadraticOracle(np.eye(5), np.arange(5))
    >> x_opt, message, history = newton(oracle, np.zeros(5), line_search_options={'method': 'Constant', 'c': 1.0})
    >> print('Found optimal point: {}'.format(x_opt))
       Found optimal point: [ 0.  1.  2.  3.  4.]
    """
    history = defaultdict(list) if trace else None
    line_search_tool = get_line_search_tool(line_search_options)
    x_k = np.copy(x_0)

    if display:
        print('Optimization debug information')

    t_start = time()

    g = lambda t: oracle.grad(t)
    f = lambda t: oracle.func(t)
    g0 = g(x_0)
    t_elapsed = time() - t_start
    if trace:
        history['time'] = [t_elapsed]
        history['func'] = [f(x_k)]
        history['grad_norm'] = [np.linalg.norm(g0)]
        if x_k.size <= 2:
            history['x'] = [x_k]

    alpha0 = 1
    for _ in range(max_iter):
        if np.linalg.norm(g(x_k)) ** 2 <= tolerance * np.linalg.norm(g0) ** 2:
            return x_k, 'success', history
        try:
            grad = g(x_k)
            hess = oracle.hess(x_k)
            d_k = cho_solve((cho_factor(hess)), -grad)
        except LinAlgError:
            return x_k, 'newton_direction_error', history

        if not (np.all(np.isfinite(x_k)) and np.all(np.isfinite(d_k))):
            return x_k, 'computational_error', history

        alpha = line_search_tool.line_search(oracle=oracle, x_k=x_k, d_k=d_k, previous_alpha=optimal_alpha(x_k, d_k))
        x_k = x_k + alpha * d_k
        t_elapsed = time() - t_start

        if trace:
            history['time'].append(t_elapsed)
            history['func'].append(f(x_k))
            history['grad_norm'].append(np.linalg.norm(g(x_k)))
            if x_k.size <= 2:
                history['x'].append(x_k)

        if np.linalg.norm(g(x_k)) ** 2 <= tolerance * np.linalg.norm(g0) ** 2:
            return x_k, 'success', history

    return x_k, 'iterations_exceeded', history















