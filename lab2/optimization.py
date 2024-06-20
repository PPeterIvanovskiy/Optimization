import numpy as np
from collections import defaultdict
from utils import get_line_search_tool
import time
from time import time
import scipy

def conjugate_gradients(matvec, b, x_0, tolerance=1e-4, max_iter=None, trace=False, display=False):
    """
    Solves system Ax=b using Conjugate Gradients method.

    Parameters
    ----------
    matvec : function
        Implement matrix-vector product of matrix A and arbitrary vector x
    b : 1-dimensional np.array
        Vector b for the system.
    x_0 : 1-dimensional np.array
        Starting point of the algorithm
    tolerance : float
        Epsilon value for stopping criterion.
        Stop optimization procedure and return x_k when:
         ||Ax_k - b||_2 <= tolerance * ||b||_2
    max_iter : int, or None
        Maximum number of iterations. if max_iter=None, set max_iter to n, where n is
        the dimension of the space
    trace : bool
        If True, the progress information is appended into history dictionary during training.
        Otherwise None is returned instead of history.
    display:  bool
        If True, debug information is displayed during optimization.
        Printing format is up to a student and is not checked in any way.

    Returns
    -------
    x_star : np.array
        The point found by the optimization procedure
    message : string
        'success' or the description of error:
            - 'iterations_exceeded': if after max_iter iterations of the method x_k still doesn't satisfy
                the stopping criterion.
    history : dictionary of lists or None
        Dictionary containing the progress information or None if trace=False.
        Dictionary has to be organized as follows:
            - history['time'] : list of floats, containing time in seconds passed from the start of the method
            - history['residual_norm'] : list of values Euclidian norms ||g(x_k)|| of the gradient on every step of the algorithm
            - history['x'] : list of np.arrays, containing the trajectory of the algorithm. ONLY STORE IF x.size <= 2
    """
    history = defaultdict(list) if trace else None
    x_k = np.copy(x_0)
    # TODO: Implement Conjugate Gradients method.
    start_time = time()
    x_k = x_k.flatten()
    b = b.flatten()
    if max_iter is None:
        max_iter = len(b)
    residual_k = matvec(x_k) - b
    direction_k = -residual_k
    norm_b = np.linalg.norm(b)

    if trace:
        history['time'].append(0)
        history['residual_norm'].append(np.linalg.norm(residual_k))
        if len(x_k) <= 2:
            history['x'].append(np.copy(x_k))

    for iteration in range(max_iter):

        if np.linalg.norm(residual_k) <= tolerance * norm_b:
            return x_k, 'success', history

        alpha_k = (residual_k @ residual_k) / (matvec(direction_k) @ direction_k)
        x_k += alpha_k * direction_k

        new_residual_k = matvec(x_k) - b
        beta_k = (new_residual_k @ new_residual_k) / (residual_k @ residual_k)
        direction_k = -new_residual_k + beta_k * direction_k
        residual_k = new_residual_k

        if trace:
            elapsed_time = time() - start_time
            history['time'].append(elapsed_time)
            history['residual_norm'].append(np.linalg.norm(residual_k))
            if len(x_k) <= 2:
                history['x'].append(np.copy(x_k))

    if display:
        for key, value in history.items():
            print(f"{key}: {value}")
    return x_k, 'iterations_exceeded', history


def lbfgs(oracle, x_0, tolerance=1e-4, max_iter=500, memory_size=10,
          line_search_options=None, display=False, trace=False):
    """
    Limited-memory Broyden–Fletcher–Goldfarb–Shanno's method for optimization.

    Parameters
    ----------
    oracle : BaseSmoothOracle-descendant object
        Oracle with .func() and .grad() methods implemented for computing
        function value and its gradient respectively.
    x_0 : 1-dimensional np.array
        Starting point of the algorithm
    tolerance : float
        Epsilon value for stopping criterion.
    max_iter : int
        Maximum number of iterations.
    memory_size : int
        The length of directions history in L-BFGS method.
    line_search_options : dict, LineSearchTool or None
        Dictionary with line search options. See LineSearchTool class for details.
    display : bool
        If True, debug information is displayed during optimization.
        Printing format is up to a student and is not checked in any way.
    trace:  bool
        If True, the progress information is appended into history dictionary during training.
        Otherwise None is returned instead of history.

    Returns
    -------
    x_star : np.array
        The point found by the optimization procedure
    message : string
        'success' or the description of error:
            - 'iterations_exceeded': if after max_iter iterations of the method x_k still doesn't satisfy
              the stopping criterion.
    history : dictionary of lists or None
        Dictionary containing the progress information or None if trace=False.
        Dictionary has to be organized as follows:
            - history['func'] : list of function values f(x_k) on every step of the algorithm
            - history['time'] : list of floats, containing time in seconds passed from the start of the method
            - history['grad_norm'] : list of values Euclidian norms ||g(x_k)|| of the gradient on every step of the algorithm
            - history['x'] : list of np.arrays, containing the trajectory of the algorithm. ONLY STORE IF x.size <= 2
    """
    history = defaultdict(list) if trace else None
    line_search_tool = get_line_search_tool(line_search_options)
    # TODO: Implement L-BFGS method.
    # Use line_search_tool.line_search() for adaptive step size.
    x_current = x_0.copy()
    s_history, y_history, rho_history = [], [], []
    start_time = time()
    initial_gradient = oracle.grad(x_current)
    stopping_criterion = tolerance * np.linalg.norm(initial_gradient) ** 2

    if trace:
        history['time'].append(0)
        history['func'].append(oracle.func(x_0))
        history['grad_norm'].append(np.linalg.norm(oracle.grad(x_current)))
        if len(x_current) <= 2:
            history['x'].append(np.copy(x_current))

    for _ in range(max_iter):
        grad_current = oracle.grad(x_current)
        if np.linalg.norm(grad_current) ** 2 <= stopping_criterion:
            break

        search_direction = compute_direction(grad_current, s_history, y_history, rho_history)

        step_size = line_search_tool.line_search(oracle=oracle, x_k=x_current, d_k=search_direction,
                                                 previous_alpha=None)
        step = step_size * search_direction
        x_current = x_current + step
        update_histories(oracle, x_current, step, search_direction, grad_current, s_history, y_history, rho_history,
                         memory_size)

        if trace:
            elapsed_time = time() - start_time
            history['time'].append(elapsed_time)
            history['func'].append(oracle.func(x_current))
            history['grad_norm'].append(np.linalg.norm(oracle.grad(x_current)))
            if x_current.size <= 2:
                history['x'].append(np.copy(x_current))
    else:
        return x_current, 'iterations_exceeded', history

    return x_current, 'success', history


def compute_direction(grad_current, s_history, y_history, rho_history):
    if len(s_history) == 0:
        return -grad_current

    q = grad_current.copy()
    alpha_history = [0] * len(s_history)
    for i in range(len(s_history) - 1, -1, -1):
        alpha_history[i] = rho_history[i] * np.dot(s_history[i], q)
        q -= alpha_history[i] * y_history[i]

    r = q * (np.dot(s_history[-1], y_history[-1]) / np.dot(y_history[-1], y_history[-1]))

    for i in range(len(s_history)):
        beta = rho_history[i] * np.dot(y_history[i], r)
        r += s_history[i] * (alpha_history[i] - beta)

    return -r


def update_histories(oracle, x_current, step, search_direction, grad_current, s_history, y_history, rho_history,
                     memory_size):
    if len(s_history) >= memory_size and memory_size > 0:
        s_history.pop(0)
        y_history.pop(0)
        rho_history.pop(0)

    s_k = step
    y_k = oracle.grad(x_current) - grad_current
    s_history.append(s_k)
    y_history.append(y_k)
    dot_product = np.dot(y_k, s_k)
    rho_history.append(1 / dot_product if dot_product != 0 else float('inf'))


def hessian_free_newton(oracle, x_0, tolerance=1e-4, max_iter=500,
                        line_search_options=None, display=False, trace=False):
    """
    Hessian Free method for optimization.

    Parameters
    ----------
    oracle : BaseSmoothOracle-descendant object
        Oracle with .func(), .grad() and .hess_vec() methods implemented for computing
        function value, its gradient and matrix product of the Hessian times vector respectively.
    x_0 : 1-dimensional np.array
        Starting point of the algorithm
    tolerance : float
        Epsilon value for stopping criterion.
    max_iter : int
        Maximum number of iterations.
    line_search_options : dict, LineSearchTool or None
        Dictionary with line search options. See LineSearchTool class for details.
    display : bool
        If True, debug information is displayed during optimization.
        Printing format is up to a student and is not checked in any way.
    trace:  bool
        If True, the progress information is appended into history dictionary during training.
        Otherwise None is returned instead of history.

    Returns
    -------
    x_star : np.array
        The point found by the optimization procedure
    message : string
        'success' or the description of error:
            - 'iterations_exceeded': if after max_iter iterations of the method x_k still doesn't satisfy
              the stopping criterion.
    history : dictionary of lists or None
        Dictionary containing the progress information or None if trace=False.
        Dictionary has to be organized as follows:
            - history['func'] : list of function values f(x_k) on every step of the algorithm
            - history['time'] : list of floats, containing time in seconds passed from the start of the method
            - history['grad_norm'] : list of values Euclidian norms ||g(x_k)|| of the gradient on every step of the algorithm
            - history['x'] : list of np.arrays, containing the trajectory of the algorithm. ONLY STORE IF x.size <= 2
    """
    history = defaultdict(list) if trace else None
    line_search_tool = get_line_search_tool(line_search_options)
    x_k = np.copy(x_0).reshape(-1)

    # TODO: Implement hessian-free Newton's method.
    # Use line_search_tool.line_search() for adaptive step size.

    def hessian_vector_product(vector):
        return oracle.hess_vec(x_k, vector)

    step_size = 1
    gradient_k = oracle.grad(x_k)
    initial_grad_norm = np.linalg.norm(gradient_k)
    start_time = time()
    stopping_criterion = tolerance * initial_grad_norm ** 2

    if trace:
        history['time'].append(0)
        history['func'].append(oracle.func(x_0))
        history['grad_norm'].append(initial_grad_norm)
        if len(x_k) <= 2:
            history['x'].append(np.copy(x_k))

    for _ in range(max_iter):
        if np.linalg.norm(gradient_k) ** 2 <= stopping_criterion:
            break

        theta = min(0.5, np.sqrt(np.linalg.norm(gradient_k)))
        direction_k = conjugate_gradients(hessian_vector_product, -gradient_k, -gradient_k, tolerance=theta)[0]
        while gradient_k @ direction_k >= 0:
            theta /= 10
            direction_k = conjugate_gradients(hessian_vector_product, -gradient_k, direction_k, tolerance=theta)[0]

        x_k = x_k + step_size * direction_k
        step_size = line_search_tool.line_search(oracle=oracle, x_k=x_k, d_k=direction_k, previous_alpha=step_size)
        gradient_k = oracle.grad(x_k)

        if trace:
            elapsed_time = time() - start_time
            history['time'].append(elapsed_time)
            history['func'].append(oracle.func(x_k))
            history['grad_norm'].append(np.linalg.norm(oracle.grad(x_k)))
            if x_k.size <= 2:
                history['x'].append(x_k)
    else:
        return x_k, 'iterations_exceeded', history

    if display:
        for key, value in history.items():
            print(f"{key}: {value}")

    return x_k, 'success', history



def gradient_descent(oracle, x_0, tolerance=1e-5, max_iter=10000,
                     line_search_options=None, trace=False, display=False):
    """
    Gradien descent optimization method.

    Parameters
    ----------
    oracle : BaseSmoothOracle-descendant object
        Oracle with .func(), .grad() and .hess() methods implemented for computing
        function value, its gradient and Hessian respectively.
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
        Printing format and is up to a student and is not checked in any way.

    Returns
    -------
    x_star : np.array
        The point found by the optimization procedure
    message : string
        "success" or the description of error:
            - 'iterations_exceeded': if after max_iter iterations of the method x_k still doesn't satisfy
                the stopping criterion.
            - 'computational_error': in case of getting Infinity or None value during the computations.
    history : dictionary of lists or None
        Dictionary containing the progress information or None if trace=False.
        Dictionary has to be organized as follows:
            - history['time'] : list of floats, containing time in seconds passed from the start of the method
            - history['func'] : list of function values f(x_k) on every step of the algorithm
            - history['grad_norm'] : list of values Euclidian norms ||g(x_k)|| of the gradient on every step of the algorithm
            - history['x'] : list of np.arrays, containing the trajectory of the algorithm. ONLY STORE IF x.size <= 2

    Example:
    --------
    >> oracle = QuadraticOracle(np.eye(5), np.arange(5))
    >> x_opt, message, history = gradient_descent(oracle, np.zeros(5), line_search_options={'method': 'Armijo', 'c1': 1e-4})
    >> print('Found optimal point: {}'.format(x_opt))
       Found optimal point: [ 0.  1.  2.  3.  4.]
    """
    history = defaultdict(list) if trace else None
    line_search_tool = get_line_search_tool(line_search_options)
    x_k = np.copy(x_0)

    # TODO: Implement gradient descent
    # Use line_search_tool.line_search() for adaptive step size.
    message = 'Yes!'
    start_time = time() if trace else None

    for i in range(max_iter):
        grad_x_k = oracle.grad(x_k)
        grad_norm = np.linalg.norm(grad_x_k)

        if grad_norm ** 2 <= tolerance:
            break

        d_k = -grad_x_k

        alpha = line_search_tool.line_search(oracle, x_k, d_k, previous_alpha=None)
        if alpha is None or not np.isfinite(alpha):
            message = 'compute_error'
            break

        x_k += alpha * d_k

        if trace:
            history['time'].append(time() - start_time)
            history['func'].append(oracle.func(x_k))
            history['grad_norm'].append(grad_norm)
            if len(x_k) <= 2:
                history['x'].append(np.copy(x_k))

        if display:
            print(f'итерация {i}, прошло времени: {time() - start_time:.2f}s, градиентная норма: {grad_norm:.2e}')

    else:
        message = 'iterations_exceeded'

    return x_k, message, history