import inspect
import pandas as pd
import numpy as np
from numpy.linalg import norm
from scipy.optimize import line_search, brent
from functools import partial
from copy import deepcopy


def get_output(func, x_new, dataset_rec, flag,  dataset=None):
    """
    Generates output data for functions.
    """
    res = {'point': None, 'value_func': None, 'report': None,
           'interim_results_dataset': None}
    args_func = inspect.getfullargspec(func)[0]
    shape_arg = len(args_func)
    col = [f'x_{i}' for i in range(1, shape_arg+1)]
    res['point'] = x_new
    res['value_func'] = func(*x_new)
    if dataset_rec:
        res['interim_results_dataset'] = pd.DataFrame(dataset,
                                                      columns=col+['f'])
    res['report'] = flag
    return res


def _find_lr_rate(lr, func, diff_func, x):
    """
    Returns the value of the learning rate for the fastest descent.
    Used for GDSteepest.
    """
    return func(*[x[i] - lr*diff_func(*x)[i] for i in range(len(x))])


def GD(func, diff_func, lr_rate=0.1, x_old=None, accuracy=10**-5, maxiter=500,
       interim_results=False, dataset_rec=False):
    """
    Returns dict with the minimum of a function using the gradient descent,
    value of function, report and intermediate results in pandas.DataFrame
    (opt). Given the function and the derivative of the function, return the
    minimum of the function with the specified accuracy.
    Parameters
    ----------
    func : callable ``f(x)``
        Objective function to be minimized.
    diff_func : callable ``f'(x)``
        Derivative of the objective function.
    lr_rate : float
        Learning rate or step.
    x_old: list
        Starting point
    accuracy : float, optional
        stop criterion.
    maxiter : int
        Maximum number of iterations to perform.
    interim_results : bool, optional
        If True, print intermediate results.
    dataset_rec : bool, optional
        If True, an entry in pandas.DataFrame intermediate results.
    Examples
    --------
    >>> from HW3.gradient import *
    >>> def f(x, y):
            return x**2 + 2*y
    >>> def pf(x, y):
            return np.array([2*x, 2])
    >>> minimum = GD(f, pf, lr_rate=0.05)
    >>> print(minimum['point'])
    [1.3220708194808046e-23, -49.000000000000426]
    """
    flag = None
    dataset = []
    iterat = 0
    args_func = inspect.getfullargspec(func)[0]
    shape_arg = len(args_func)
    col = [f'x_{i}' for i in range(1, shape_arg+1)]
    if x_old is None:
        x_old = [1] * shape_arg
    crit_stop = accuracy*2
    while crit_stop > accuracy and iterat < maxiter:
        x_new = [x_old[i] - lr_rate * diff_func(*x_old)[i]
                 for i in range(shape_arg)]
        crit_stop = norm([x_new[i] - x_old[i] for i in range(shape_arg)])
        x_old = x_new
        if dataset_rec:
            dataset.append([*x_new, func(*x_new)])
        if interim_results:
            print(f'''{iterat}:
            point = {x_new}    f(point) = {func(*x_new)}''')
        iterat += 1
    if crit_stop <= accuracy and flag != 2:
        flag = 0
    elif iterat == maxiter and flag != 2:
        flag = 1
    return get_output(func, x_new, dataset_rec, flag, dataset)


def GDSS(func, diff_func, lr_rate=0.1, e=0.1, d=0.5, x_old=None, maxiter=500,
         accuracy=10**-5, interim_results=False, dataset_rec=False):
    """
    Returns dict with the minimum of a function using the gradient descent with
    step splitting, value of function, report and intermediate results in
    pandas.DataFrame (opt). Given the function and the derivative of the
    function, return the minimum of the function with the specified accuracy.
    Parameters
    ----------
    func : callable ``f(x)``
        Objective function to be minimized.
    diff_func : callable ``f'(x)``
        Derivative of the objective function.
    lr_rate : float
        Learning rate or step.
    e : float
        The value of the evaluation parameter.
    d : float
        The value of the crushing parameter.
    x_old: list
        Starting point
    accuracy : float, optional
        stop criterion.
    maxiter : int
        Maximum number of iterations to perform.
    interim_results : bool, optional
        If True, print intermediate results.
    dataset_rec : bool, optional
        If True, an entry in pandas.DataFrame intermediate results.
    Examples
    --------
    >>> from HW3.gradient import *
    >>> def f(x, y):
            return x**2 + 2*y
    >>> def pf(x, y):
            return np.array([2*x, 2])
    >>> minimum = GDSS(f, pf, lr_rate=0.05)
    >>> print(minimum['point'])
    [1.3220708194808046e-23, -49.000000000000426]
    """
    res = {'point': None, 'value_func': None, 'report': None,
           'interim_results_dataset': None}
    flag = None
    dataset = []
    iterat = 0
    args_func = inspect.getfullargspec(func)[0]
    shape_arg = len(args_func)
    col = [f'x_{i}' for i in range(1, shape_arg+1)]
    if x_old is None:
        x_old = [1] * shape_arg
    crit_stop = accuracy*2
    while crit_stop > accuracy and iterat < maxiter:
        x_new = [x_old[i] - lr_rate * diff_func(*x_old)[i]
                 for i in range(shape_arg)]
        if func(*x_new) > func(*x_old) - e*lr_rate*norm(diff_func(*x_old)):
            lr_rate *= d
        crit_stop = norm([x_new[i] - x_old[i] for i in range(shape_arg)])
        x_old = x_new
        if dataset_rec:
            dataset.append([*x_new, func(*x_new)])
        if interim_results:
            print(f'''{iterat}:
            point = {x_new}    f(point) = {func(*x_new)}''')
        iterat += 1
    if crit_stop <= accuracy and flag != 2:
        flag = 0
    elif iterat == maxiter and flag != 2:
        flag = 1
    return get_output(func, x_new, dataset_rec, flag, dataset)


def GDSteepest(func, diff_func, x_old=None, accuracy=10**-5, maxiter=500,
               interim_results=False, dataset_rec=False):
    """
    Returns dict with the minimum of a function using the steepest gradient
    descent, value of function, report and intermediate results in
    pandas.DataFrame (opt). Given the function and the derivative of the
    function, return the minimum of the function with the specified accuracy.
    Parameters
    ----------
    func : callable ``f(x)``
        Objective function to be minimized.
    diff_func : callable ``f'(x)``
        Derivative of the objective function.
    x_old: list
        Starting point
    accuracy : float, optional
        stop criterion.
    maxiter : int
        Maximum number of iterations to perform.
    interim_results : bool, optional
        If True, print intermediate results.
    dataset_rec : bool, optional
        If True, an entry in pandas.DataFrame intermediate results.
    Examples
    --------
    >>> from HW3.gradient import *
    >>> def f(x, y):
            return x**2 + 2*y
    >>> def pf(x, y):
            return np.array([2*x, 2])
    >>> minimum = GDSteepest(f, pf)
    >>> print(minimum['point'])
    [0.9999999853676476, -998.9999999854012]
    """
    flag = None
    dataset = []
    iterat = 0
    args_func = inspect.getfullargspec(func)[0]
    shape_arg = len(args_func)
    col = [f'x_{i}' for i in range(1, shape_arg+1)]
    if x_old is None:
        x_old = [1] * shape_arg
    crit_stop = accuracy*2
    while crit_stop > accuracy and iterat < maxiter:
        lr_rate = brent(partial(_find_lr_rate,
                                func=func, diff_func=diff_func, x=x_old),
                        brack=(0, 1))
        x_new = [x_old[i] - lr_rate * diff_func(*x_old)[i]
                 for i in range(shape_arg)]
        crit_stop = norm([x_new[i] - x_old[i] for i in range(shape_arg)])
        x_old = x_new
        if dataset_rec:
            dataset.append([*x_new, func(*x_new)])
        if interim_results:
            print(f'''{iterat}:
            point = {x_new}    f(point) = {func(*x_new)}''')
        iterat += 1
    if crit_stop <= accuracy and flag != 2:
        flag = 0
    elif iterat == maxiter and flag != 2:
        flag = 1
    return get_output(func, x_new, dataset_rec, flag, dataset)


def newton_algorithm(func, func_diff, func_diff_2x, func_diff_xy,
                     lamda=1, max_steps=500, accuracy=10**-5):
    '''
    Returns tuple with the minimum of a function using the newton algorithm with conjugate 
    gradient and the value of the point in the extremum points.
    Parameters
    ----------
    func : callable ``f(x)``
        Objective function to be minimized.
    func_diff : callable ``f'(x)``
        Derivative of the objective function.
    func_diff_2x: callable ``f''(x)``
        The double derivative of the objective function by one variable.
    func_diff_xy: callable ``f''(x, y)``
        The double derivative of the objective function by x and y variables.
    lamda: int
        Algorithm step.
    accuracy : float, optional
        stop criterion.
    max_steps : int
        Maximum number of iterations to perform.
    Examples
    --------
    >>> from HW3.gradient import *
    >>> def f(x, y):
            return x**2 + 2*y
    >>> def f_diff(x, y):
            return np.array([2*x, 2])
    >>> def f_diff_2x(x, y):
            return np.array([2, 0])
    >>> def f_diff_xy(x, y):
            return np.array([2, 2])
    >>> m = newton_algorithm(f, f_diff, f_diff_2x, f_diff_xy)
    >>> print(m)
    (array([[0.],
        [0.]]),
     0.0)
    """
    '''
    args_func = inspect.getfullargspec(func)[0]
    shape_arg = len(args_func)
    ones = [1]*shape_arg
    first_diff = func_diff(*ones).reshape((shape_arg, 1))
    second_diff_2x = func_diff_2x(*ones).reshape((shape_arg, 1))
    second_diff_xy = func_diff_xy(*ones).reshape((shape_arg, 1))
    

    x_old = np.ones((shape_arg, 1))
    d2_f = np.zeros((shape_arg,) * 2)
    for i in range(shape_arg):
        d2_f[i][i] = second_diff_2x[i]
        d2_f[i][i-1] = second_diff_xy[i]
        
    d2_f_inv = np.linalg.inv(d2_f)
    
    steps = 0
    crit_stop = deepcopy(accuracy)
    while steps < max_steps and crit_stop <= accuracy:
        x_new = x_old - lamda*np.dot(d2_f_inv, first_diff)
        crit_stop = norm([list(x_new.reshape(1, x_new.size)[0])[i] - list(x_old.reshape(1, x_old.size)[0])[i] for i in range(shape_arg)])
        print(crit_stop)
        x_old = deepcopy(x_new)
        steps += 1
        
    return x_new, func(*list(x_new.reshape(1, x_new.size)[0]))

