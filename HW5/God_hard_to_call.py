from scipy.optimize import minimize
import numpy as np
from sympy import *


def eq_dual_newton(func: str, equality: list, x0: tuple, tol=5):
    """
    Solving an optimisation problem for a function with equality-type
    constraints by Newton's method (a way of solving the dual problem).
    Returns the found point and the value of the function in it.
    Parameters
    ----------
    func : string
        Function for optimisation.
    equality : list
        List of strings with given linear constraints.
    x0 : tuple
        Starting point.
    tol : int, default=5
        The number of numbers after the point for rounding.
    Examples
    --------
    >>> from HW5.God_hard_to_call import *
    >>> func = '(x1-2)**2 + x2**2'
    >>> eqs = ['x1+x2-4 =0', '-4-2*x1+x2 =0']
    >>> x, y = eq_dual_newton(func, eqs, (0, 0))
    >>> x, y
    (array([-0.,  4.]), 20.0000000000000)
    """
    try:
        func = sympify(func)
        equality = [sympify(eq.partition('=')[0]) for eq in equality]
    except SympifyError:
        print('Неверно заданы функции')
    func_c = lambda x: func.subs(dict(zip(func.free_symbols, x)))
    eq_func = lambda x: [us.subs(dict(zip(us.free_symbols, x))) for us in equality]
    eq_constraints = {'type': 'eq',
                      'fun': eq_func}
    res = minimize(func_c, x0, method='SLSQP', constraints=eq_constraints)
    return res['x'].round(tol), round(res['fun'], tol)
