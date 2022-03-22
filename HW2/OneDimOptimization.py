from scipy import optimize
import sympy as sp
import warnings
import pandas as pd
import numpy as np
from sympy import *
from copy import deepcopy
from scipy.optimize import brentq
from sympy.plotting import plot
from sympy.plotting.plot import MatplotlibBackend, Plot


def golden_ratio(func, search_area, extreme_type='min', accuracy=10**(-5),
                 maxiter=500, interim_results=False, dataset_rec=False):
    """
    Returns dict with the minimum or maximum of a function of one variable on 
    the segment [a, b] using the golden ratio method, value of function, report
    and intermediate results in pandas.DataFrame (optional).
    Given a function of one variable and a possible bracketing interval,
    return the minimum or maximum of the function isolated to a fractional
    precision of accuracy.
    Parameters
    ----------
    func : str
        Objective function to minimize or maximize.
    search_area : tuple or list
        [a, b], where (a<b) – the interval within which the maximum
        and minimum are searched.
    accuracy : float, optional
        x tolerance stop criterion.
    maxiter : int
        Maximum number of iterations to perform.
    interim_results : bool, optional
        If True, print intermediate results.
    dataset_rec : bool, optional
        If True, an entry in pandas.DataFrame intermediate results.
    Examples
    --------
    >>> from HW2.OneDimOptimization import golden_ratio
    >>> minimum = golden_ratio(func='x**2', search_area=(1, 2))
    >>> print(minimum['point'])
    1.0000048224378428
    """
    str_error = 'Длина отрезка должна быть больше заданной точности'
    assert search_area[1] - search_area[0] > accuracy, str_error
    res = {'point': None, 'value_func': None, 'report': None,
           'interim_results_dataset': None}
    df = pd.DataFrame(columns=['a', 'b', 'x1', 'x2', 'f1', 'f2'])
    x = sp.Symbol('x')
    try:
        func = eval(func)
    except NameError:
        print('Функция должна быть задана через x')
        return None
    proportion = 1.6180339887
    a, b = search_area
    iterat = 0
    while abs(b - a) >= accuracy and iterat != maxiter:
        iterat += 1
        x1 = b - (b-a) / proportion
        x2 = a + (b-a) / proportion
        f1, f2 = float(func.subs({x: x1})), float(func.subs({x: x2}))
        if dataset_rec:
            d = {'a': a, 'b': b, 'x1': x1, 'x2': x2, 'f1': f1, 'f2': f2}
            df = df.append(d, ignore_index=True)

        if extreme_type == 'min':
            if f1 >= f2:
                a = x1
            else:
                b = x2
        elif extreme_type == 'max':
            if f1 <= f2:
                a = x1
            else:
                b = x2

        if interim_results:
            print(f'''{iterat}:
            a = {a}, b = {b},
            x1 = {x1}, x2 = {x2},
            f1 = {f1}, f2 = {f2}''')

    if dataset_rec:
        res['interim_results_dataset'] = df
    res['point'] = (a+b) / 2
    res['value_func'] = func.subs({x: res['point']})
    if abs(b-a) <= accuracy:
        res['report'] = 0
    elif iter == maxiter:
        res['report'] = 1
    else:
        res['report'] = 2
    return res
  
  
def get_sympy_subplots(plot: Plot):
    backend = MatplotlibBackend(plot)

    backend.process_series()
    backend.fig.tight_layout()
    return backend.fig, backend.ax[0]
  
 
def parabola_method(func: str,
                    limits: list,
                    type_extr = 'min',
                    accuracy: float=10**(-5),
                    max_iterations: int=500,
                    intermediate_results: bool=False,
                    intermediate_writing: bool=False,
                    figure=True):
    """
    Search for the extremum of a function of one variable using the parabola method.
    
    args:
        mandatory:
            - func - function in analytical form;
            - limits - optimization area boundaries;
        optional:
            - accuracy - optimization precision by argument (default: 10^-5);
            - max_iterations - maximum number of iterations (default: 500;
            - intermediate_results - flag "output intermediate results" (default: False);
            - intermediate_writing - flag "writing intermediate results to dataset" (default: False);
            
    outputs:
        - The found value of the extremum point coordinate;
        - Function value at the extremum point;
        - Algorithm report;
    """
    try:
        X = symbols('X')
        F = eval(func)

        if type_extr == 'max':
            F = F * (-1)

        left_hand, right_hand = eval(limits)

        left_hand = float(left_hand)
        right_hand = float(right_hand)

        if left_hand > right_hand:
            left_hand, right_hand = right_hand, left_hand

        results = pd.DataFrame(columns=['x1', 'x2', 'x3', 'f1', 'f2', 'f3'])
        iteration_num = 0
        d = 100
        while iteration_num < max_iterations:
            # need to satisfy inequality x1<x2<x3, f(x1) >= f(x2) <= f(x3) 

            if not(d > accuracy) and d != 0.0:
                if intermediate_writing:
                    if intermediate_results:
                        print(f"iteration_num = {iteration_num}; x1 = {x1}; x2 = {x2}; x3 = {x3}; f1 = {f1}; f2 = {f2}; f3 = {f3}")
                    if intermediate_writing:
                        results = results.append({'x1': x1, 'x2': x2, 'x3': x3, 'f1': f1, 'f2': f2, 'f3': f3}, ignore_index=True)
                print(results)
                if figure:
                    f_res = F.subs(x, x_res)
                    p = plot(func, show=False)
                    fig, axe = get_sympy_subplots(p)
                    axe.plot(x_res, f_res, "o", c='red')
                    fig.show()
                return x_res, f_res, 'Значение с заданной точностью', d

            # first step
            if iteration_num == 0:
                x1 = left_hand
                x3 = right_hand
                x2 = (x1+x3)/2


            f1 = F.subs(X, x1)
            f2 = F.subs(X, x2)
            f3 = F.subs(X, x3)


            # second step
            # first  formula - a0 = f1, a1 = (f2-f1)/(x2-x1), a2 = 1/(x3-x2) * ((f3-f1)/(x3-x1) - (f2-f1)/(x2-x1))
            # second formula - x_ = 1/2*(x1 + x2 - a1/a2)
            a0 = float(deepcopy(f1))
            a1 = float((f2-f1)/(x2-x1))
            a2 = float(1/(x3-x2) * ((f3-f1)/(x3-x1) - (f2-f1)/(x2-x1)))


            if iteration_num > 0:
                x_old = deepcopy(x_)

            x_ = 1/2*(x1 + x2 - a1/a2)


            # check num of step - go to step 4 if its first step
            if iteration_num > 0:
                d = abs(x_old - x_)

                if d <= accuracy:
                    x_res = deepcopy(x_)


            # step 4
            f_x_ = F.subs(X, x_)

            # step 5
            # suppose x1 = x_ = ..., f1 = f_x_ = ...
            if x1 < x_ < x2 < x3 and f_x_ >= f2: # x* in [x_, x3]
                x1 = deepcopy(x_)
                f1 = deepcopy(f_x_)
            elif x1 < x2 < x_ < x3 and f2 >= f_x_: # x* in [x2, x3]
                x1 = deepcopy(x2)
                f1 = deepcopy(f2)
                x2 = deepcopy(x_)
                f2 = deepcopy(f_x_)

            iteration_num += 1
            if intermediate_results:
                print(f"iteration_num = {iteration_num}; x1 = {x1}; x2 = {x2}; x3 = {x3}; f1 = {f1}; f2 = {f2}; f3 = {f3}")
            if intermediate_writing:
                results = results.append({'x1': x1, 'x2': x2, 'x3': x3, 'f1': f1, 'f2': f2, 'f3': f3}, ignore_index=True)

        f_res = F.subs(X, x_res)

        if intermediate_writing:
            print(results)

        if figure:
            p = plot(func, show=False)
            fig, axe = get_sympy_subplots(p)
            axe.plot(x_res, f_res, "o", c='red')
            fig.show()

        return x_res, f_res, 'Достигнуто максимальное количество итераций'
    except:
        return 'Выполнено с ошибкой'
  
def BrantMethod(func: str,
                limits: list,
                accuracy: float=10**(-500),
                max_iterations: int=500,
                intermediate_results: bool=False,
                intermediate_writing: bool=False,
                figure=True):
    
    X = symbols('x')
    F = eval(func)
    
    
    a, b = eval(limits)
    r = (3-5**(1/2))/2
    
    
    x = a + r*(b - a)
    w = a + r*(b - a)
    v = a + r*(b - a)
    

    
    f_x = F.subs(X, x)
    f_w = F.subs(X, w)
    f_v = F.subs(X, v)
    
    
    d_cur = b - a
    d_prv = b - a
    
    results = pd.DataFrame(columns=['x', 'w', 'v', 'f_x', 'f_w', 'f_v'])
    step_num = 0
    while step_num <= max_iterations:
        if max(abs(x - a), abs(b - x)) < accuracy:
            
            if intermediate_results:
                print(f"iteration_num = {iteration_num}; x = {x}; w = {w}; v = {v}; f_x = {f_x}; f_w = {f_w}; f_v = {f_v}")
            if intermediate_writing:
                results = results.append({'x': x, 'w': w, 'v': v, 'f_x': f_x, 'f_w': f_w, 'f_v': f_v}, ignore_index=True)
            if intermediate_writing:
                print(results)

            if figure:
                p = plot(func, show=False)
                fig, axe = get_sympy_subplots(p)
                axe.plot(x, F.subs(X, x), "o", c='red')
                fig.show()
            
            return x, F.subs(X, x), step_num
        g = d_prv /2
        d_prv = deepcopy(d_cur)
#         res = solve([f_x - a0*x**2 - a1*x - a2, f_w - a0*w**2 - a1*w -a2, f_v - a0*v**2 - a1*v - a2],
#                          [a0, a1, a2], dict=True)[0]
#         u = -res[a1]/(2*res[a0])
        u = parabola_method(func, limits, figure=False)[1]
        if u is None or not(a <= u <= b) or abs(u-x) > g:
            if x < (a+b)/2:
                u = x + r*(b-x)
                d_prv = b - x
            else:
                u = x - r*(x - a)
                d_prv = x - a
        d_cur = abs(u - x)
        if F.subs(X, u) > f_x:
            if u < x:
                a = deepcopy(u)
            else:
                b = deepcopy(u)
            if F.subs(X, u) <= f_w or w == x:
                v = deepcopy(w)
                w = deepcopy(u)
            else:
                if F.subs(X, u) <= f_v or v == x or v == w:
                    v = deepcopy(u)
        else:
            if u < x:
                b = deepcopy(x)
            else:
                a = deepcopy(x)
            v = deepcopy(w)
            w = deepcopy(x)
            x = deepcopy(u)
        step_num += 1
        
        if intermediate_results:
            print(f"iteration_num = {iteration_num}; x = {x}; w = {w}; v = {v}; f_x = {f_x}; f_w = {f_w}; f_v = {f_v}")
        if intermediate_writing:
            results = results.append({'x': x, 'w': w, 'v': v, 'f_x': f_x, 'f_w': f_w, 'f_v': f_v}, ignore_index=True)
    
    if intermediate_writing:
        print(results)
        
    if figure:
        p = plot(func, show=False)
        fig, axe = get_sympy_subplots(p)
        axe.plot(x, F.subs(X, x), "o", c='red')
        fig.show()
    
    return x, F.subs(X, x), step_num
  
def bfgs(func, diff_func, x0, extreme_type='min', accuracy=10**-5, maxarg=100,
         firstW=10**-4, secondW=0.1, maxiter=500, interim_results=False,
         dataset_rec=False):
    """
    Minimize or maximize a function using the BFGS algorithm.
    Parameters
    ----------
    func : callable ``f(x)``
        Objective function to be minimized or maximized.
    diff_func : callable ``f'(x)``
        Objective function gradient.
    x0 : float, int
        Starting point.
    extreme_type : str, optional
        Maximizing or minimizing.
    accuracy : float, optional
        Gradient norm (abs) in x0 must be less than `accuracy` before successful
        termination.
    maxarg : float, int, optional
        Maximum value of the argument function.
    firstW : int or ndarray, optional
        Parameter for Armijo condition rule.
    secondW : callable, optional
        Parameter for curvature condition rule.
    maxiter : int, optional
        Maximum number of iterations to perform.
    interim_results : bool, optional
        If True, print intermediate results.
    dataset_rec : bool, optional
        If True, an entry in pandas.DataFrame intermediate results.
    Examples
    --------
    >>> from HW2.OneDimOptimization import bfgs
    >>> def f(x):
    ...     return -x/(x**2+2)
    ... def f1(x):
    ...     return 2*x**2/(x**2+2)**2 - 1/(x**2+2)
    >>> bfgs(f, f1, 1, maxiter=5000, extreme_type='max')
    {'point': -1.4142509823545781,
     'value_func': 0.35355339046951084,
     'report': 1,
     'interim_results_dataset': None}
    >>> bfgs(f, f1, 1, maxiter=5000)
    {'point': 1.4142144676627526,
     'value_func': -0.35355339059320134,
     'report': 1,
     'interim_results_dataset': None}
    """
    if extreme_type == 'max':
        f = lambda x: -func(x)
        diff_f = lambda x: -diff_func(x)
    res = {'point': None, 'value_func': None, 'report': None,
           'interim_results_dataset': None}
    dataset = []
    iterat = 0
    if extreme_type == 'max':
        g = diff_f(x0)
    else:
        g = diff_func(x0)
    Hk = 1  # is initial approximation
    xk = x0
    while abs(g) > accuracy and iterat < maxiter and xk < maxarg:
        if dataset_rec:
            dataset.append([xk, func(xk), g, Hk])
        if interim_results:
            print(f'''{iterat}:
            xk = {xk}    f(xk) = {func(xk)}    g = {g}    Hk = {Hk}''')
        pk = -Hk * g
        try:
            if extreme_type == 'max':
                line_search = optimize.line_search(f, diff_f, np.array(xk),
                                                   np.array(pk), c1=firstW,
                                                   c2=secondW)
            else:
                line_search = optimize.line_search(func, diff_func, np.array(xk),
                                                   np.array(pk), c1=firstW,
                                                   c2=secondW)
            if line_search[0]:
                alpha_k = line_search[0]
            else:
                print('не смогли найти лучшее приближение')
                res['report'] = 4
                break
        except optimize.linesearch.LineSearchWarning:
            print('не смогли найти лучшее приближение')
            res['report'] = 4
            break
        xkp = xk + alpha_k * pk
        sk = xkp - xk
        xk = xkp

        if extreme_type == 'max':
            g2 = diff_f(xk)
        else:
            g2 = diff_func(xkp)
        yk = g2 - g
        g = g2

        iterat += 1
        rho = 1.0 / (yk * sk)
        A1 = 1 - rho * sk * yk
        A2 = 1 - rho * yk * sk
        Hk = A1 * Hk * A2 + (rho * sk**2)


    res['point'] = xk
    res['value_func'] = func(xk)
    if dataset_rec:
        res['interim_results_dataset'] = pd.DataFrame(dataset,
                                                      columns=['xk', 'f', 'g', 'Hk'])
    if abs(g) <= accuracy and res['report'] != 4:
        res['report'] = 1
    elif iterat == maxiter and res['report'] != 4:
        res['report'] = 2
    elif xk >= maxarg and res['report'] != 4:
        res['report'] = 3
    return res
