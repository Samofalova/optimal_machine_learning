import itertools
import random
import warnings
import inspect
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from functools import partial
from copy import deepcopy
from sympy import *
import scipy.linalg as la
from numpy.linalg import norm
from sympy.plotting.plot import MatplotlibBackend, Plot
from sympy.plotting import plot3d, plot
from scipy import optimize
from scipy.optimize import brent, minimize, linprog
from sklearn.linear_model import (LinearRegression, Lasso, Ridge,
                                  LogisticRegression)
from sklearn.preprocessing import PolynomialFeatures
from sympy.core.numbers import NaN
from sklearn.svm import LinearSVC, SVC
get_ipython().run_line_magic('matplotlib', 'notebook')


def get_sympy_subplots(plot: Plot):
    """
    Allows you to combine plot3d graphics from sympy and plot from matplotlib
    return fig and ax for graph
    """
    backend = MatplotlibBackend(plot)
    backend.process_series()
    backend.fig.tight_layout()
    return backend.fig, backend.ax[0]


def get_data():
    """
    Initial function for data entry
    return data with variables and math function
    """
    data = dict()
    text = 'Введите названия переменных (x y): '
    data['X'] = symbols(input(text).split())
    assert len(data['X']) == 2, 'переменные заданы неверно'

    f = input('Введите функцию (y*x+2): ')
    data['func'] = Matrix([f])

    data['limit'] = int(input('Есть ли ограничения? (1 – да, 0 – нет): '))
    if data['limit']:
        str_x = 'Введите ограничения для x [-10, 10]: '
        str_y = 'Введите ограничения для y [-10, 10]: '
        try:
            data['x_min'], data['x_max'] = eval(input(str_x))
            data['y_min'], data['y_max'] = eval(input(str_y))
        except ValueError:
            raise Exception('ограничения заданы неверно')
        range_x = 'границы для ограничения x перепутаны'
        range_y = 'границы для ограничения y перепутаны'
        assert data['x_min'] < data['x_max'], range_x
        assert data['y_min'] < data['y_max'], range_y

    return data


def get_crit(func: Matrix, X: list):
    """
    func: sympy.Matrix(['x + y']),
    X: [sympy.Symbol('x'), sympy.Symbol('y')]
    return critical points
    """
    gradf = simplify(func.jacobian(X))
    return solve(gradf, X, dict=True)


def filter_point(point: list, x_min, x_max, y_min, y_max):
    """
    point: [(1, 2), (2, 3)] – list of tuples, critical points for filtering
    x_min, x_max, y_min, y_max – int or float, constraints for variables
    """
    x, y = point.values()
    if simplify(x).is_real and simplify(y).is_real:
        if x_min <= x <= x_max and y_min <= y <= y_max:
            return True
    return False


def type_point(func, X, x0):
    """
    func: sympy.Matrix(['x + y']),
    X: [sympy.Symbol('x'), sympy.Symbol('y')],
    x0: (1, 2) – tuple of int or float numbers, critical point
    return type of critical points
    """
    hessianf = simplify(hessian(func, X))
    H = np.array(hessianf.subs(dict(zip(X, x0)))).astype('float')
    l, v = la.eig(H)
    if(np.all(np.greater(l, np.zeros(2)))):
        return 'minimum'
    elif(np.all(np.less(l, np.zeros(2)))):
        return 'maximum'
    else:
        return 'saddle'


def get_extremums():
    """
    returns a tuple from the source data and the results of the function.
    data: dict - dictionary of source data, stores the name of variables,
    function, constraints.
    points: list – a list of tuples, each element stores a point,
    the value of the function at the point and the type of extremum.
    """
    data = get_data()
    crit = get_crit(data['func'], data['X'])
    if data['limit']:
        print('Если в списке критических точек есть комплексные числа, мы их не выводим.')
        f = partial(filter_point, x_min=data['x_min'], x_max=data['x_max'], 
                                  y_min=data['y_min'], y_max=data['y_max'])
        crit = list(filter(f, crit))
    if len(crit) > 40:
        n = int(input('Точек больше 40, сколько вывести? '))
        crit = crit[:n]
    points = []
    for x in crit:
        if len(x) == 2:
            x1, x2 = x.values()
            z = data['func'].subs(x)[0]
            try:
                type_x = type_point(data['func'], data['X'], x.values())
                points.append(((x1, x2), z, type_x))
            except (ValueError, TypeError):
                points.append((x, 'crit point'))
                continue 
        else:
            points.append((x, 'crit point'))

    return data, points


def show_chart(data: dict, points:list):
    """
    data: dictionary with variables and function
    points: list with points
    """
    p = plot3d(data['func'][0], show=False)
    fig, axe = get_sympy_subplots(p)
    if points:
        x1, x2, x3 = [], [], []
        for point in points:
            if point[-1] != 'crit point':
                x1.append(point[0][0])
                x2.append(point[0][1])
                x3.append(float(point[1]))
        axe.scatter(x1, x2, x3, "o", color='red', zorder=3)
    fig.show()


def main():
    """
    The main function that solves the equation and outputs the graph
    """
    try:
        data, points = get_extremums()
    except NotImplementedError: 
        return 'Для данного выражения нет аналитического решения'
    print(*points, sep='\n')
    show_chart(data, points)


def golden_ratio(func, search_area, extreme_type='min', accuracy=10 ** (-5),
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
    x = Symbol('x')
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
        x1 = b - (b - a) / proportion
        x2 = a + (b - a) / proportion
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
    res['point'] = (a + b) / 2
    res['value_func'] = func.subs({x: res['point']})
    if abs(b - a) <= accuracy:
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


def center_point(F, x1, x2, x3):
    if (x1, F(x1)) == (x2, F(x2)) or (x2, F(x2)) == (x3, F(x3)) or (x1, F(x1)) == (x3, F(x3)):
        return None
    f_1, f_2, f_3 = F(x1), F(x2), F(x3)

    a1 = (f_2 - f_1) / (x2 - x1)
    a2 = 1 / (x3 - x2) * ((f_3 - f_1) / (x3 - x1) - (f_2 - f_1) / (x2 - x1))

    point = 0.5 * (x1 + x2 - a1 / a2)
    return point


def parabola_method(func: str,
                    limits: list,
                    type_extr='min',
                    accuracy: float = 10 ** (-5),
                    max_iterations: int = 500,
                    intermediate_results: bool = False,
                    intermediate_writing: bool = False,
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

            if not (d > accuracy) and d != 0.0:
                if intermediate_writing:
                    if intermediate_results:
                        print(
                            f"iteration_num = {iteration_num}; x1 = {x1}; x2 = {x2}; x3 = {x3}; f1 = {f1}; f2 = {f2}; f3 = {f3}")
                    if intermediate_writing:
                        results = results.append({'x1': x1, 'x2': x2, 'x3': x3, 'f1': f1, 'f2': f2, 'f3': f3},
                                                 ignore_index=True)
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
                x2 = (x1 + x3) / 2

            f1 = F.subs(X, x1)
            f2 = F.subs(X, x2)
            f3 = F.subs(X, x3)

            # second step
            # first  formula - a0 = f1, a1 = (f2-f1)/(x2-x1), a2 = 1/(x3-x2) * ((f3-f1)/(x3-x1) - (f2-f1)/(x2-x1))
            # second formula - x_ = 1/2*(x1 + x2 - a1/a2)
            a0 = float(deepcopy(f1))
            a1 = float((f2 - f1) / (x2 - x1))
            a2 = float(1 / (x3 - x2) * ((f3 - f1) / (x3 - x1) - (f2 - f1) / (x2 - x1)))

            if iteration_num > 0:
                x_old = deepcopy(x_)

            x_ = 1 / 2 * (x1 + x2 - a1 / a2)

            # check num of step - go to step 4 if its first step
            if iteration_num > 0:
                d = abs(x_old - x_)

                if d <= accuracy:
                    x_res = deepcopy(x_)

            # step 4
            f_x_ = F.subs(X, x_)

            # step 5
            # suppose x1 = x_ = ..., f1 = f_x_ = ...
            if x1 < x_ < x2 < x3 and f_x_ >= f2:  # x* in [x_, x3]
                x1 = deepcopy(x_)
                f1 = deepcopy(f_x_)
            elif x1 < x2 < x_ < x3 and f2 >= f_x_:  # x* in [x2, x3]
                x1 = deepcopy(x2)
                f1 = deepcopy(f2)
                x2 = deepcopy(x_)
                f2 = deepcopy(f_x_)

            iteration_num += 1
            if intermediate_results:
                print(
                    f"iteration_num = {iteration_num}; x1 = {x1}; x2 = {x2}; x3 = {x3}; f1 = {f1}; f2 = {f2}; f3 = {f3}")
            if intermediate_writing:
                results = results.append({'x1': x1, 'x2': x2, 'x3': x3, 'f1': f1, 'f2': f2, 'f3': f3},
                                         ignore_index=True)

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
                accuracy: float = 10 ** (-5),
                max_iterations: int = 500,
                intermediate_results: bool = False,
                intermediate_writing: bool = False):
    r = (3 - 5 ** (1 / 2)) / 2

    a, b = limits

    x = a + r * (b - a)
    w = a + r * (b - a)
    v = a + r * (b - a)

    d_cur = a - b
    d_prv = a - b

    results = pd.DataFrame(columns=['step_num', 'a', 'b', 'x', 'w', 'v', 'u'])
    step_num = 0
    while max(x - a, b - x) > accuracy:

        if step_num >= max_iterations:
            if intermediate_results:
                print(f'step_num: {step_num}, a: {a}, b: {b}, x: {x}, w: {w}, v: {w}, u: {u}')
            if intermediate_writing:
                print(result)

            return x, func(x), 'Достигнуто макисмальное количество итераций'

        g = d_prv / 2
        d_prv = d_cur
        u = center_point(F, x, w, v)
        if not u or (u < a or u > b) or abs(u - x) > g:
            if x < (a + b) / 2:
                u = x + r * (b - x)
                d_prv = b - x
            else:
                u = x - r * (x - a)
                d_prv = (x - a)
        d_cur = abs(u - x)

        if func(u) > func(x):
            if u < x:
                a = u
            else:
                b = u
            if func(u) <= func(w) or w == x:
                v = w
                w = u
            else:
                if func(u) <= func(v) or v == x or v == w:
                    v = u
        else:
            if u < x:
                b = x
            else:
                a = x
            v = w
            w = x
            x = u
        step_num += 1
        if intermediate_results:
            print(f'step_num: {step_num}, a: {a}, b: {b}, x: {x}, w: {w}, v: {w}, u: {u}')
        if intermediate_writing:
            results = results.append({'step_num': step_num, 'a': a, 'b': b,
                                      'x': x, 'w': w, 'v': v, 'u': u}, ignore_index=True)

    if intermediate_writing:
        print(result)

    return x, func(x), 'Достигнута заданная точность'


def bfgs(func, diff_func, x0, extreme_type='min', accuracy=10 ** -5, maxarg=100,
          firstW=10 ** -4, secondW=0.1, maxiter=500, interim_results=False,
          dataset_rec=False):
    '''
    '''
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
                line_search = optimize.line_search(f, diff_f, np.array(xk), np.array(pk),
                                                   c1=firstW, c2=secondW)
            else:
                line_search = optimize.line_search(func, diff_func, np.array(xk), np.array(pk),
                                                   c1=firstW, c2=secondW)
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
        Hk = A1 * Hk * A2 + (rho * sk ** 2)
    #         Hk *= (1 - 2*rho*yk)**2  # выведенная формула c помощью матричных вычислений
    #         Hk *= np.sqrt((1 - 2*rho*yk)**2)

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


class InsufficientData(Exception):
    """
    Rises when there is not enough data to build a model.
    """
    def __str__(self):
        string_exp = 'Данных недостаточно. Их должно быть не менее 2^k строк, \
где k – количество признаков. Если признак 1, то хотя бы 10 строк.'
        return string_exp


class LinearlyDependent(Exception):
    """
    Rises when there is a linear relationship between the signs, which makes it impossible
    to apply the method of least squares.
    """
    def __str__(self):
        return 'Присутствуют линейно зависимые признаки. Мы не можем применить МНК.'


class DegreeError(Exception):
    """
    Rises when an incorrect degree is entered to construct a polynomial regression.
    """
    def __str__(self):
        return 'Степень полинома должна быть целым неотрицательным числом.'


class NegativeValue(Exception):
    """
    Rises when negative values of y have been fed to the input for the exponential regression. 
    """
    def __str__(self):
        return 'Значения y должны быть положительными'


class VeryBig(Exception):
    """
    Rises when the free term in the exponential regression is too large and further calculations 
    are impossible. 
    """
    def __str__(self):
        return 'Свободный член получился слишком большим, чтобы произвести вычисления'


class RegularizationError(Exception):
    """
    It raises when one tries to apply regularization to polynomial regression. 
    """
    def __str__(self):
        return 'К сожалению, мы не можем построить полиномиальную регрессию с регулязацией'


def student_del(X, y):
    """
    Excludes points from the data according to Student's regularization.
    """
    X_new = X.copy()
    y_new = y.copy()
    for line in range(len(X)):
        if len(X.shape) == 2:
            if X_new.drop(index=line).var().sum() < X_new.var().sum():
                X_new = X_new.drop(index=line)
                y_new = y_new.drop(index=line)
        else:
            if X_new.drop(index=line).var() < X_new.var():
                X_new = X_new.drop(index=line)
                y_new = y_new.drop(index=line)
    return X_new, y_new


def check_data(X):
    """
    Checks the data size for sufficiency to build an adequate model.
    """
    if len(X.shape) == 2 and X.shape[1] > 1:
        if X.shape[0] < 2**X.shape[1] or len(X) < 10:
            raise InsufficientData
    else:
        if len(X) < 10:
            raise InsufficientData


def plot_3d_regression(X, y, coef, a0, reg_type):
    """
    Plots the graph of the function and plots points from the dataset. 
    Works if there are two feature. 
    Parameters
    ----------
    X : pandas.DataFrame
        A dataset of features.
    y : numpy.array or pandas.DataFrame
        A dataset of targets.
    coef : numpy.array
        The coefficients at the features in the resulting function.
    a0: float or numpy.float64
        The free term in the resulting function.
    reg_type : string
        Type of function.
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel("X0")
    ax.set_ylabel("X1")
    ax.set_zlabel("y")

    if reg_type=='lin':
        ax.scatter(X.iloc[:, 0], X.iloc[:, 1], y, marker='.', color='red') 

        a = np.arange(min(X.min())-1, max(X.max())+1)
        xs = np.tile(a,(len(a),1))
        ys = np.tile(a, (len(a),1)).T       
        zs = a0 + coef[0]*xs + ys * coef[1]

    elif reg_type=='exp':
        ax.scatter(X.iloc[:, 0], X.iloc[:, 1], y, marker='.', color='red') 

        a = np.arange(min(X.min())-1, max(X.max())+1)
        xs = np.tile(a,(len(a),1))
        ys = np.tile(a, (len(a),1)).T  
        zs = a0 * np.exp(coef[0]*xs) * np.exp(ys * coef[1])
    elif reg_type=='poly1': 
        ax.scatter(X.iloc[:, 1], X.iloc[:, 2], y, marker='.', color='red') 
        a = np.arange(min(X.iloc[:, 1:].min())-1, max(X.iloc[:, 1:].max())+1)
        xs = np.tile(a,(len(a),1))
        ys = np.tile(a, (len(a),1)).T 
        zs = a0 + coef[0]*xs + ys * coef[1]
    else:
        ax.scatter(X.iloc[:, 1], X.iloc[:, 2], y, marker='.', color='red') 
        a = np.arange(X.iloc[:, 1].min(), X.iloc[:, 1].max())
        xs = np.tile(a,(len(a),1))
        ys = np.square(np.tile(a, (len(a),1))).T 
        zs = a0 + coef[0]*xs + xs**2 * coef[1]      

    ax.plot_surface(xs, ys, zs, alpha=0.5)
    plt.show()


def plot_2d_regression(X, y, coef, a0, reg_type):
    """
    Plots the graph of the function and plots points from the dataset. 
    Works if there are one feature. 
    Parameters
    ----------
    X : pandas.DataFrame
        A dataset of features.
    y : numpy.array or pandas.DataFrame
        A dataset of targets.
    coef : numpy.array
        The coefficients at the features in the resulting function.
    a0: float or numpy.float64
        The free term in the resulting function.
    reg_type : string
        Type of function.
    """
    xs = np.linspace(X.min()-1, X.max()+1)
    if reg_type == 'lin':
        zs = a0 + xs*coef
    else:
        zs = a0 * np.exp(coef*xs)
    plt.plot(xs, zs, color="blue", linewidth=3, label='Прогноз')
    plt.scatter(X, y, marker='.', color='red', label='Исходные') 
    plt.legend()
    plt.show()


def exp_regression(X, y, tol=5, regularization=None, alpha=1.0, draw=False):
    """
    Ordinary least squares exponential regression. Fits the model to minimize the residual 
    sum of squares between the observed targets of the data set and the targets predicted 
    by the approximation.
    Parameters
    ----------
    X : pandas.DataFrame
        A dataset of features.
    y : pandas.DataFrame
        A dataset of targets.
    tol : int, default=
        The number of decimal places to round the coefficient when writing the function 
        in analytic form.
    regularization: string, optional
        Type of regularization.
    alpha : float, default=1.0
        Constant for regularization.
    draw : bool, optional
        Flag for the chart. If the value is True, the graph is drawn. Works only for 
        two- and three-dimensional cases.
    Examples
    --------
    >>> from HW4.regression import *
    >>> import pandas as pd
    >>> import yfinance as yf
    >>> aapl = yf.download('AAPL', '2021-01-01', '2022-01-01')
    >>> aapl = aapl.reset_index(level=0)
    >>> exp_regression(aapl['Volume'], aapl['Close'])
    {'func': '143.6968 * exp(-0.0*x0)',
     'weights': -2.697485449323164e-10,
     'bias': 143.6967960345703}
    """
    if not (y > 0).all():
        raise NegativeValue
    y_new = np.log(y)
    check_data(X)
    if len(X.shape) < 2:
        X = X.to_numpy().reshape(-1, 1)

    if regularization is None:
        if X.shape[1] >= 2 and np.linalg.det(X.T@X) == 0:
            raise LinearlyDependent
        reg = LinearRegression().fit(X, y_new)
    elif regularization == 'L1':
        reg = Lasso(alpha=alpha).fit(X, y_new)
    elif regularization == 'L2':
        reg = Ridge(alpha=alpha).fit(X, y_new)
    elif regularization == 'Student':
        X_new, y_log_new = student_del(pd.DataFrame(X), 
                                       pd.DataFrame(y_new))
        check_data(X_new)
        if len(X_new.shape) < 2:
            X_new = X_new.to_numpy().reshape(-1, 1)
        reg = LinearRegression().fit(X_new, y_log_new)
    X = pd.DataFrame(X)
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings('error')
            if regularization == 'Student':
                weights, bias = reg.coef_[0], np.exp(reg.intercept_)[0]
            else:
                weights, bias = reg.coef_, np.exp(reg.intercept_)
    except RuntimeWarning:
        raise VeryBig
    if X.shape[1] >= 2:
        func = f'{round(bias, tol)}'
        k = len(weights)
        for i in range(k):
            coef = weights[i]
            func = func + f' * exp({round(coef, tol)}*x{k-i})'
    else:
        weights = weights[0]
        bias = bias
        func = f'{round(bias, tol)}' + f' * exp({round(weights, tol)}*x0)'
    if draw == True and X.shape[1] > 2:
        print('К сожалению, мы не можем построить график, так как размерность пространства признаков велика.')
    elif draw == True and X.shape[1] == 2:
        plot_3d_regression(X, y, weights, bias, reg_type='exp')
    elif draw == True and X.shape[1] == 1:
        plot_2d_regression(X, y, weights, bias, reg_type='exp')
    return {'func': func, 
            'weights': weights, 
            'bias': bias}


def poly_regression(X, y, degree, tol=5, regularization=None, alpha=1.0, draw=False):
    """
    Ordinary least squares polynomial regression. Fits the model to minimize the residual 
    sum of squares between the observed targets of the data set and the targets predicted 
    by the approximation.
    Parameters
    ----------
    X : pandas.DataFrame
        A dataset of features.
    y : pandas.DataFrame
        A dataset of targets.
    tol : int, default=
        The number of decimal places to round the coefficient when writing the function 
        in analytic form.
    regularization: string, optional
        Type of regularization.
    alpha : float, default=1.0
        Constant for regularization.
    draw : bool, optional
        Flag for the chart. If the value is True, the graph is drawn. Works only for 
        two- and three-dimensional cases.
    Examples
    --------
    >>> from HW4.regression import *
    >>> import pandas as pd
    >>> import yfinance as yf
    >>> aapl = yf.download('AAPL', '2021-01-01', '2022-01-01')
    >>> aapl = aapl.reset_index(level=0)
    >>> poly_regression(aapl['Volume'], aapl['Close'], degree=1)
    {'func': '143.4969 + -0.0*x1',
     'weights': -2.805183057424643e-08,
     'bias': 143.49689552425573}
    """

    if degree <= 0 or type(degree)!=int: 
        raise DegreeError

    check_data(X)
    if len(X.shape) < 2:
        X = X.to_numpy().reshape(-1, 1)
    poly = PolynomialFeatures(degree)
    X_poly = poly.fit_transform(X)
    if regularization is not None and degree>1:
        raise RegularizationError

    if X.shape[1] >= 2 and np.linalg.det(X.T@X) == 0:
        raise LinearlyDependent
    reg = LinearRegression().fit(X_poly, y)


    if regularization == 'Student':
        weights, bias = reg.coef_[0], reg.intercept_[0]
    else:
        weights, bias = reg.coef_, reg.intercept_    

    X_poly = pd.DataFrame(X_poly)

    weights = weights[1:]

    if X.shape[1]==1 and degree==1 : 
        weights = weights[0] 
        func = f'{round(bias, tol)} + {round(weights, tol)}*x1'
    elif X.shape[1]==1 and degree>1: 
        func = f'{round(bias, tol)}' 
        for i in range(1, degree+1):
            func = func + f' + {round(weights[i-1], tol)}*x1^{i}'
    elif X.shape[1]>=2 and degree==1:
        func = f'{round(bias, tol)}' 
        for i in range(len(weights)):
            func = func + f' + {round(weights[i], tol)}*x{i+1}'
    else: 
        func = 'К сожалению, мы не можем вывести функцию для множественной полиномиальной регрессии'

    if draw == True and X.shape[1] == 1 and degree==1: 
        plot_2d_regression(X, y, weights, bias, reg_type='lin')
    elif draw==True and (X.shape[1]==2 and degree==1):    
        plot_3d_regression(X_poly, y, weights, bias, reg_type='poly1')
    elif draw==True and (X.shape[1]==1 and degree==2):
        plot_3d_regression(X_poly, y, weights, bias, reg_type='poly2')
    else:
        print('К сожалению, мы не можем построить график, так как размерность пространства признаков велика.')

    return {'func': func, 
            'weights': weights, 
            'bias': bias}


def lin_regression(X, y, tol = 5, regularization = None, alpha=1.0, draw = False):
    """
    Ordinary least squares linear regression. Fits the model to minimize the residual 
    sum of squares between the observed targets of the data set and the targets predicted 
    by the approximation.
    Parameters
    ----------
    X : pandas.DataFrame
        A dataset of features.
    y : pandas.DataFrame
        A dataset of targets.
    tol : int, default=
        The number of decimal places to round the coefficient when writing the function 
        in analytic form.
    regularization: string, optional
        Type of regularization.
    alpha : float, default=1.0
        Constant for regularization.
    draw : bool, optional
        Flag for the chart. If the value is True, the graph is drawn. Works only for 
        two- and three-dimensional cases.
    Examples
    --------
    >>> from HW4.regression import *
    >>> import pandas as pd
    >>> import yfinance as yf
    >>> aapl = yf.download('AAPL', '2021-01-01', '2022-01-01')
    >>> aapl = aapl.reset_index(level=0)
    >>> lin_regression(aapl[['Open', 'Volume']], aapl['Close'], regularization='L2')
    {'func': '0.33756 + 1.00283x1 -0.0x2',
     'weights': array([ 1.00283393e+00, -6.79315538e-09]),
     'bias': 0.33756283615903726}
    """
    y_new = y.to_numpy()
    check_data(X)

    if len(X.shape) < 2:
        X = X.to_numpy().reshape(-1, 1)

    if regularization is None:
        reg = LinearRegression().fit(X, y_new)
    elif regularization == 'L1':
        reg = Lasso(alpha=alpha).fit(X, y_new)
    elif regularization == 'L2':
        reg = Ridge(alpha=alpha).fit(X, y_new)

    elif regularization == 'Student':
        X_new, y_new = student_del(pd.DataFrame(X), 
                                       pd.DataFrame(y_new))
        check_data(X_new)
        if len(X_new.shape) < 2:
            X_new = X_new.to_numpy().reshape(-1, 1)
        reg = LinearRegression().fit(X_new, y_new)

    if regularization == 'Student':
        weights, bias = reg.coef_[0], reg.intercept_[0]
    else:
        weights, bias = reg.coef_, reg.intercept_
    func = str(round(bias, tol)) + ' '
    for i in range(len(weights)):
        if str(weights[i])[0] == '-':
            func += str(round(weights[i], tol)) + 'x' + str(i + 1) + ' '
        else:
            func += '+ ' + str(round(weights[i], tol)) + 'x' + str(i + 1) + ' '
    if draw == True and X.shape[1] > 2:
        print('К сожалению, мы не можем построить график, так как размерность пространства признаков велика.')
    elif draw == True and X.shape[1] == 2:
        plot_3d_regression(X, y, weights, bias, reg_type='lin')
    elif draw == True and X.shape[1] == 1:
        plot_2d_regression(X, y, weights, bias, reg_type='lin')
    return {'func': func[:-1], 
            'weights': weights, 
            'bias': bias}


def log_barriers(func: str, restrictions: list, start_point: tuple = tuple(), accuracy:float = 10**(-6), max_steps: int=500):
    '''
    Solving an optimisation problem for a function with equality-type
    constraints by log barriers method (a way of solving the dual problem).
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
    >>> from HW5.log_barriers import *
    >>> func = '(x1-2)**2 + x2**2'
    >>> eqs = ['x1+x2-4 >= 0', '-4-2*x1+x2 >= 0']
    >>> x, y = eq_dual_newton(func, eqs, (0, 0))
    >>> x, y
    (array([-0.,  4.]), 20.0000000000000)
    '''
    tao = 1
    v = 10
    for i in range(len(restrictions)):
        if '>' in restrictions[i]:
            restrictions[i] = restrictions[i][:restrictions[i].index('>')].replace(' ', '')
        else:
            restrictions[i] = restrictions[i][:restrictions[i].index('<')].replace(' ', '')

    phi = f'{tao}*({func})'
    for exp in restrictions:
        phi += f' - log({exp})'

    X = Matrix([sympify(phi)])
    symbs = list(sympify(phi).free_symbols)

    if len(start_point) == 0:
        start_point = first_phase(restrictions, symbs)
    if start_point == 'Введенные ограничения не могут использоваться вместе':
        return start_point
    elif start_point == 'Невозможно подобрать внутреннюю точку для данного метода':
        return start_point

    try:
        res = sympify(func).subs(list(zip(symbs, start_point)))
    except:
        return 'Введена первоначальная точка, которая не удовлетворяет неравенствам'
    Y = Matrix(list(symbs))

    df = X.jacobian(Y).T
    ddf = df.jacobian(Y)

    lst_for_subs =  list(zip(symbs, start_point))
    dfx0 = df.subs(lst_for_subs)
    ddfx0 = ddf.subs(lst_for_subs)

    xk = ddfx0.inv() @ dfx0
    next_point = [start_point[i]-xk[i] for i in range(len(start_point))]
    tao = tao*v


    res_new = sympify(func).subs(list(zip(symbs, next_point)))
    if type(res_new) == NaN:
        return np.array(start_point), res

    steps = 1
    while abs(res_new - res) > accuracy and max_steps > steps:
        phi = f'{tao}*({func})'
        for exp in restrictions:
            phi += f' - log({exp})'

        X = Matrix([sympify(phi)])
        symbs = list(sympify(phi).free_symbols)
        Y = Matrix(list(symbs))

        df = X.jacobian(Y).T
        ddf = df.jacobian(Y)

        lst_for_subs =  list(zip(symbs, start_point))
        dfx0 = df.subs(lst_for_subs)
        ddfx0 = ddf.subs(lst_for_subs)

        xk = ddfx0.inv() @ dfx0
        old_point = deepcopy(next_point)
        next_point = [next_point[i]-xk[i] for i in range(len(next_point))]
        res = deepcopy(res_new)
        res_new = sympify(func).subs(list(zip(symbs, next_point)))
        if type(res_new) == NaN:
            return np.array(old_point), res

        tao = tao*v
        steps += 1

    return np.array(next_point), res_new


def first_phase(restrictions: list, symbs: list) -> tuple:

    s = 1000
    restrictions_sympy = [sympify(i) for i in restrictions]
    res_functions = []
    for i in range(100, -100, -1):
        if s >= 0:
            x = [i for j in range(len(symbs))]
            s = max([expr.subs(list(zip(symbs, x))) for expr in restrictions_sympy])

    if s < 0:
        return x
    elif s > 0:
        return 'Введенные ограничения не могут использоваться вместе'
    elif s == 0:
        return 'Невозможно подобрать внутреннюю точку для данного метода'


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


def get_pain(first_line, second_line, third_line, A):
    """
    Auxiliary function for matrix recording.
    Parameters
    ----------
    first_line : list
        The first row for the matrix.
    second_line : list
        The second row for the matrix.
    third_line : list
        The third row for the matrix.
    A : list
        A list of coefficients for variables in a constraint 
        of the equality type.
    """
    pain = []
    for line in [first_line, second_line, third_line]:
        if line == third_line and not A:
            continue
        pain_ = []
        pain_.append([l[i] for l in line for i in range(l.shape[1])]) 
        pain_.append([l[i] for l in line for i in range(l.shape[1], len(l))])
        pain.extend(pain_)
    return pain


def get_first_line(F0, Fi, Y, A, n, us_n):
    """
    Calculates the first row for the matrix.
    """
    first_line = [F0.jacobian(Y).T.jacobian(Y)]
    s = Matrix(np.zeros(shape=(n, n)))
    for u_n in us_n:
        s += Matrix([u_n]).jacobian(Y).jacobian(Y)
    first_line[0] += s
    first_line.append(Fi.jacobian(Y).T)
    if A:
        first_line.append(A.T)
    return first_line


def get_second_line(diag_lam, Fi, Y, A, m, n, p):
    """
    Calculates the second row for the matrix.
    """
    second_line = [diag_lam@Fi.jacobian(Y),
                   diag(*Fi, unpack=True)]
    if A:
        second_line.append(Matrix(np.zeros((m, (n+m+p-2*n)))))
    return second_line


def get_right_pain(F0, Y, Fi, lmbd, A, mu, diag_lam, e, Axb):
    """
    Calculates the right-hand side of a system of linear 
    equations from a terrible condition.
    """
    pain1 = [F0.jacobian(Y).T + Fi.jacobian(Y).T@lmbd]
    if A:
        pain1[0] = pain1[0] + A.T@mu
    pain2 = [diag_lam@Fi + e]
    pain3 = [Axb]
    pain = []
    pain.extend(pain1)
    pain.extend(pain2)
    if A:
        pain.extend(pain3)
    right_pain = Matrix(pain)
    return right_pain


def search_point(x0, lmbd, mu, var,
                 left_pain, right_pain, 
                 n, m, p, Fi, func, tol, A):
    """
    Searches for a point for the specified conditions.
    """
    diff_f = 1000
    point_old = x0
    point = x0

    Lmbd = np.array([1 for i in lmbd.free_symbols])
    MU = np.array([1 for i in mu.free_symbols])

    fear_left = dict(zip(var + list(lmbd.free_symbols),
                    list(point) + list(Lmbd)))
    fear_right = dict(zip(var + list(lmbd.free_symbols) + list(mu.free_symbols),
                    list(point) + list(Lmbd) + list(MU)))

    while diff_f >= tol:
        left_pain_inv = left_pain.subs(fear_right).inv()
        right_pain_new = -1*right_pain.subs(fear_right)
        horror = left_pain_inv @ right_pain_new
        horror = horror.subs(fear_right)

        if A:
            dx, dl, dm  = [horror[i: i+n] for i in range(0, horror.shape[0], n)]
        else:
            dx, dl = [horror[i: i+n] for i in range(0, horror.shape[0], n)]
        alpha_p = 1
        point_old = point
        point = np.array(point_old) + alpha_p * np.array(dx)
        fear_right = dict(zip(var + list(lmbd.free_symbols) + list(mu.free_symbols),
                              list(point) + list(Lmbd) + list(MU)))
        panic = sum(1 for el in Fi.subs(fear_right) if el <= 0)
        while panic < Fi.shape[0]:
            alpha_p *= 0.5
            point = np.array(point_old) + alpha_p * np.array(dx)
            point = point.subs(fear_right)
            fear_right = dict(zip(var + list(lmbd.free_symbols) + list(mu.free_symbols),
                              list(point) + list(Lmbd) + list(MU)))
            panic = sum(1 for el in Fi.subs(fear_right) if el <= 0)
        thing = []
        for i in range(len(dl)):
            if dl[i] < 0:
                thing.append(-0.9 * Lmbd[i] / dl[i])
        if thing:
            alpha_d = min(1, *thing)
        else:
            alpha_d = 1
        Lmbd = np.array([Lmbd[i] + alpha_d*dl[i] for i in range(len(Lmbd))])
        MU = 0.1 * MU  
        fear_right_old = dict(zip(var + list(lmbd.free_symbols) + list(mu.free_symbols),
                              list(point_old) + list(Lmbd) + list(MU)))
        fear_right = dict(zip(var + list(lmbd.free_symbols) + list(mu.free_symbols),
                              list(point) + list(Lmbd) + list(MU)))
        diff_f = abs(func.subs(fear_right) - func.subs(fear_right_old))
        return point, func.subs(fear_right)


def terrifying(func, x0, us, a, b, tol=10**-5):
    """
    This function is written to terrify anyone with formulas and source code.
    Solving the optimization problem for a function with inequality type 
    constraints by the direct-dual inner point method. Returns the found point 
    and the value of the function at that point.
    Parameters
    ----------
    func : string
        Function for optimisation.
    x0 : list
        Starting point.
    us : list
        A list of constraints set by strings. 
        Be sure to write as '4*x - 4 = 0' or '4*x - 4 < 0'
    a : list
        A list of coefficients for variables in a constraint 
        of the equality type.
    b : list
        List of free members for equality type constraint.
    tol : float, default=10**-5
        The criterion for stopping the point search.
    Examples
    --------
    >>> from HW5.God_hard_to_call import *
    >>> x, y = terrifying('x1 - 2*x2 ',
                          [0.5, 0.5],
                          ['-1-x1+x2^2 <=0', '-x2<=0', 'x1+x2-1=0'],
                          [[1, 1]],
                          [-1])
    >>> x, y
    (array([0.0277777777777777, 0.972222222222222], dtype=object), 
    -1.91666666666667)
    """
    try:
        func = sympify(func)
        us_eq = [sympify(u.partition('=')[0]) for u in us if '<' not in u]
        us_n = [sympify(u.partition('<')[0]) for u in us if '<' in u]
        var = list(func.free_symbols)
        F0 = Matrix([func])
        Fi = Matrix(*[us_n])
        Y = Matrix(var)
        n = len(var)  # колич переменных
        m = len(us_n)  # неравенство 
        p = len(us_eq)  # равенство
        lmbd = Matrix([f'lambda{i}' for i in range(m)])
        diag_lam = diag(*lmbd, unpack=True)
        mu = Matrix([f'mu{i}' for i in range(p)])
        v = 0.1
        A = Matrix(a)
        B = Matrix(b)
        Axb = Matrix(np.zeros(shape=B.shape))
        e = Matrix(np.ones(shape=len(us_n)))
        diag_f = diag(*Fi, unpack=True)

        first_line = get_first_line(F0, Fi, Y, A, n, us_n)
        second_line = get_second_line(diag_lam, Fi, Y, A, m, n, p)
        third_line = [A, Matrix(np.zeros((p, m))), Matrix(np.zeros((p, p)))]

        left_pain = Matrix(get_pain(first_line, second_line, third_line, A))
        right_pain = get_right_pain(F0, Y, Fi, lmbd, A, mu, diag_lam, e, Axb)

        point, y = search_point(x0, lmbd, mu, var,
                                left_pain, right_pain, 
                                n, m, p, Fi, func, tol, A)
        return point, y
    except:
        str1 = 'Мы сделали все возможное, чтобы решить эту задачу. '
        str2 = 'К сожалению, не помогла даже молитва((('
        print(str1+str2)
        raise


def logistic_regression_with_l1(x_train: np.array, y_train: np.array, x_test: np.array,
        penalty=None, C=1, l1_ratio=None, draw=False, degree=1):

    # проверка параметров
    assert C > 0, 'Сила регуляризации дожна быть больше, чем 0!'
    if l1_ratio is not None:
        assert 0 <= l1_ratio <= 1, 'Параметр l1_ration должен быть удовлетворять неравенству 0 <= l1_ratio <= 1!'
    assert degree > 0 and isinstance(degree, int), 'Параметр degree должен быть целым положительным числом!'
    assert len(np.unique(y_train)), 'В y_train должно быть больше одного класса!'

    if x_train.shape[0] < 2**x_train.shape[1]:
        print('Для оптимального результата количество наблюдений должно быть больше 2^k.')
        IsContinue = int(input('Все равно продолжить? (0/1): '))
        if not IsContinue:
            return

    poly = PolynomialFeatures(degree, include_bias=False)

    x_poly = poly.fit_transform(x_train)
    x_poly_test = poly.transform(x_test)

    if penalty == 'l1':
        penalty = 'elasticnet'
        l1_ratio = 1
    elif penalty != 'l1':
        penalty = 'elasticnet'
        if l1_ratio is None:
            l1_ratio = 0.8

    model = LogisticRegression(max_iter=50_000, penalty=penalty, solver='saga', l1_ratio=l1_ratio, C=C)
    model.fit(x_poly, y_train)
    y_pred = model.predict(x_poly_test)

    res = {'x_test': x_poly_test,
           'y_pred': y_pred,
           'intercept': model.intercept_,
           'coef': model.coef_}

    if x_test.shape[1] == 2 and draw:
        # create scatter plot for samples from each class
        for class_value in range(2):
            # get row indexes for samples with this class
            row_ix = np.where(y_pred == class_value)
            # create scatter of these samples
            plt.scatter(x_poly_test[row_ix, 0], x_poly_test[row_ix, 1])
        # show the plot
        plt.show()

    return res


def own_svm(x_train, y_train, x_test, 
        C=1, penalty=None, graph=False):

    assert C >0, "Сила регуляризации должна быть больше 0!"
    assert len(np.unique(y_train)) > 1, 'В y_train должно быть больше одного класса!'

    if x_train.shape[0] < 2**x_train.shape[1]:
        print('Для оптимального результата количество наблюдений должно быть больше 2^k.')
        IsContinue = int(input('Все равно продолжить? (0/1): '))
        if not IsContinue:
            return

    if penalty not in ['l1', 'l2']:
        penalty = 'l2'

    model = LinearSVC(max_iter=50_000, penalty=penalty, C=C, dual=False)
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)

    res = {'x_test': x_test,
           'y_pred': y_pred,
           'intercept': model.intercept_,
           'coef': model.coef_}

    if x_test.shape[1] == 2 and graph:
        for class_value in range(2):
            row_ix = np.where(y_pred == class_value)
            plt.scatter(x_test[row_ix, 0], x_test[row_ix, 1])
        plt.show()

    return res


def plot_boundary(clf, X, y, grid_step=.01, poly_featurizer=None):
    """
    Plot boundary for draw_2d.
    """
    x_min, x_max = X[:, 0].min() - .1, X[:, 0].max() + .1
    y_min, y_max = X[:, 1].min() - .1, X[:, 1].max() + .1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, grid_step),
    np.arange(y_min, y_max, grid_step))
    Z = clf.predict(poly_featurizer.transform(np.c_[xx.ravel(), yy.ravel()]))
    Z = Z.reshape(xx.shape)
    plt.contour(xx, yy, Z, cmap=plt.cm.Paired)


def draw_2d(X_test, y_hat, logit, poly):
    """
    Plot 2D-graph.
    """
    colors = ['#' + ''.join([random.choice('ABCDEF0123456789') for i in range(6)])
              for col in range(len(np.unique(y_hat)))]
    for clss in np.unique(y_hat):
        plt.scatter(X_test[y_hat == clss, 0], X_test[y_hat == clss, 1], 
                    c=colors[clss], label=clss)
    plt.xlabel("Признак 1")
    plt.ylabel("Признак 2")
    plot_boundary(logit, X_test, y_hat, grid_step=.01, poly_featurizer=poly)
    plt.legend()


def own_logistic_regression(X_train, y_train, X_test,
                            penalty='none', C=1, l1_ratio='none',
                            degree: int=1, draw=False, max_iter=100):
    """
    A function that implements a two-class classification model based on logistic regression.
    Parameters
    ----------
    X_train : numpy.array
        A set of features for the training sample.
    y_train : numpy.array
        Set of class values for the training sample.
    X_test : numpy.array
        A set of features for the test sample.
    penalty : string, default='none'
        Type of regalarisation. ['l1', 'l2', 'elasticnet', 'none']
    C : float, default=1
        Regalarisation strength.
    l1_ratio : float, default='none'
        Regalarisation strength l1-regalarisation if penalty='elfsticnet' is selected.
    degree : int, default=1
        Polynomial degree.
    draw : int, default=False
        Plotting the classification graph.
    max_iter : int, default=100
        Number of learning iterations
    Examples
    --------
    >>> from HW6.Classification_2_classes.py import *
    >>> Xn, Yn = make_blobs(n_samples=50, centers=2, n_features=2, random_state=1, cluster_std=3)
    >>> X_trainn, X_testn, y_trainn, y_testn = train_test_split(Xn, Yn, test_size=0.2)
    >>> own_logistic_regression(X_trainn, y_trainn, X_testn)
    {'X_test': array([[ -1.08681345,  10.70725528],
        [ -9.5176013 ,  -1.32484178],
        [-14.33005392,  -5.46674614],
        [ -0.70244262,   3.65837874],
        [ -9.40281334,  -3.59632261],
        [ -3.44098628,  -8.14283755],
        [ -3.72107801,   1.87087294],
        [ -7.48076226,  -1.1600423 ],
        [ -2.02823058,   1.59918157],
        [ -2.17684453,   1.77291462]]),
     'y_hat': array([0, 1, 1, 0, 1, 1, 0, 1, 0, 0]),
     'intercept': array([-1.83565162]),
     'coef': array([[-0.62953643, -0.91903067]])}
    """
    param = {'penalty': penalty, 
             'C': C, 
             'l1_ratio': l1_ratio}
    str1 = 'Не выполняется условие 0 <= l1_ratio <= 1'
    str2 = 'Не выполняется условие C > 0'
    str3 = 'Всего один класс, мы не можем обучить модель'
    assert l1_ratio=='none' or 0 <= l1_ratio <= 1, str1
    assert C > 0, str2
    assert len(np.unique(y_train)) > 1, str3
    poly = PolynomialFeatures(degree, include_bias=False)
    X_poly = poly.fit_transform(X_train)
    X_test_poly = poly.transform(X_test)
    if X_poly.shape[0] < 2**X_poly.shape[1]:
        print('Данных недостаточно для обучения коэффициентов. Продолжить?')
        answ = input('0 – Нет\n1 – Да')
        if answ == 0:
            return None
    if penalty == 'elasticnet' and l1_ratio == 'none':
        l1_ratio = 0.8
    if penalty == 'none':
        del param['l1_ratio']
        del param['C']
    if penalty not in ['elasticnet', 'none']:
        del param['l1_ratio']
    model = LogisticRegression(max_iter=max_iter,
                               solver='saga',
                               **param)
    model.fit(X_poly, y_train)
    y_hat = model.predict(X_test_poly)
    res = {'X_test': X_test_poly,
           'y_hat': y_hat,
           'intercept': model.intercept_,
           'coef': model.coef_}
    if X_poly.shape[1] == 2 and draw:
        draw_2d(X_test_poly, y_hat, model, poly)
    return res


def draw_rdf(X, y, clf):
    """
    Plot 2D-graph for SVM. 
    """
    colors = ['#' + ''.join([random.choice('ABCDEF0123456789') for i in range(6)])
              for col in range(len(np.unique(y)))]
    for clss in np.unique(y):
        plt.scatter(X[y == clss, 0], X[y == clss, 1], 
                    c=colors[clss], label=clss)
    plt.xlabel("Признак 1")
    plt.ylabel("Признак 2")
    h = .02
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contour(xx, yy, Z, cmap=plt.cm.Paired)


def logistic_regression_with_rbf(X_train, y_train, X_test,
                                 C=1, gamma='scale',
                                 draw=False, max_iter=100):
    """
    A function implementing a two-class classification model based on logistic
    regression with radial basis functions.
    Parameters
    ----------
    X_train : numpy.array
        A set of features for the training sample.
    y_train : numpy.array
        Set of class values for the training sample.
    X_test : numpy.array
        A set of features for the test sample.
    C : float, default=1
        Regalarisation strength.
    gamma : float, default='scale'
        Kernel factor.
    draw : int, default=False
        Plotting the classification graph.
    max_iter : int, default=100
        Number of learning iterations
    Examples
    --------
    >>> from HW6.Classification_2_classes.py import *
    >>> Xn, Yn = make_blobs(n_samples=50, centers=2, n_features=2, random_state=1, cluster_std=3)
    >>> X_trainn, X_testn, y_trainn, y_testn = train_test_split(Xn, Yn, test_size=0.2)
    >>> logistic_regression_with_rbf(X_trainn, y_trainn, X_testn,
                                     C=1, gamma='scale',
                                     draw=True, max_iter=100)
    {'X_test': array([[-12.26090633,  -0.19474408],
        [ -3.72107801,   1.87087294],
        [ -9.40281334,  -3.59632261],
        [ -1.53291867,   6.15493551],
        [ -0.75904895,   3.34974033],
        [ -3.44098628,  -8.14283755],
        [ -0.70244262,   3.65837874],
        [ -2.23506656,   1.74360298],
        [ -9.14095053,  -1.29792505],
        [  2.72676391,  -1.77393226]]),
    'y_hat': array([1, 0, 1, 0, 0, 1, 0, 0, 1, 0]),
    'intercept': array([-0.12607821]),
    'coef': array([[-0.32377093, -0.89114665, -0.2455269 , -1., -0.84723235,
                    0.52369094,  1.,  0.51354647,  1.,  0.27043942]])}
    """
    str1 = 'Не выполняется условие gamma > 0'
    str2 = 'Не выполняется условие C > 0'
    str3 = 'Всего один класс, мы не можем обучить модель'
    assert gamma == 'scale' or gamma > 0, str1
    assert C > 0, str2
    assert len(np.unique(y_train)) > 1, str3
    model = SVC(kernel='rbf',
                max_iter=max_iter,
                C=C,
                gamma=gamma)
    model.fit(X_train, y_train)
    y_hat = model.predict(X_test)
    res = {'X_test': X_test,
           'y_hat': y_hat,
           'intercept': model.intercept_,
           'coef': model.dual_coef_}
    if X_test.shape[1] == 2 and draw:
        draw_rdf(X_test, y_hat, model)
    return res


def svm_with_pddipm(X_train, y_train, X_test,
                    penalty='none', C=1, max_iter=100,
                    draw=False):
    """
    A function that implements a two-class classification model using
    the direct-double internal point method for the reference vector
    method training problem.
    Parameters
    ----------
    X_train : numpy.array
        A set of features for the training sample.
    y_train : numpy.array
        Set of class values for the training sample.
    X_test : numpy.array
        A set of features for the test sample.
    penalty : string, default='none'
        Type of regalarisation. ['l2', 'none']
    C : float, default=1
        Regalarisation strength.
    draw : int, default=False
        Plotting the classification graph.
    max_iter : int, default=100
        Number of learning iterations
    Examples
    --------
    >>> from HW6.Classification_2_classes.py import *
    >>> Xn, Yn = make_blobs(n_samples=50, centers=2, n_features=2, random_state=1, cluster_std=3)
    >>> X_trainn, X_testn, y_trainn, y_testn = train_test_split(Xn, Yn, test_size=0.2)
    >>> svm_with_pddipm(X_trainn, y_trainn, X_testn, max_iter=10500)
    {'X_test': array([[ -9.40281334,  -3.59632261],
        [-10.6243952 ,  -2.19347897],
        [  3.31984663,   6.63262235],
        [-12.00969936,  -2.82065719],
        [  3.57487539,   2.12286917],
        [ -0.70244262,   3.65837874],
        [ -3.72107801,   1.87087294],
        [ -8.45892304,  -4.84762705],
        [ -2.17684453,   1.77291462],
        [ -1.29908305,   6.2580992 ]]),
    'y_hat': array([1, 1, 0, 1, 0, 0, 0, 1, 0, 0]),
    'intercept': array([-1.16510368]),
    'coef': array([[-0.3087416 , -0.28134706]])}
    """
    param = {'penalty': 'l2', 
             'C': C, 
             'max_iter': max_iter}
    str1 = 'penalty может быть только l2'
    str2 = 'Не выполняется условие C > 0'
    str3 = 'Всего один класс, мы не можем обучить модель'
    assert penalty in ['none', 'l2'], str1
    assert C > 0, str2
    assert len(np.unique(y_train)) > 1, str3
    model = LinearSVC(dual=True, **param)
    model.fit(X_train, y_train)
    y_hat = model.predict(X_test)
    res = {'X_test': X_test,
           'y_hat': y_hat,
           'intercept': model.intercept_,
           'coef': model.coef_}
    if X_test.shape[1] == 2 and draw:
        draw_rdf(X_test, y_hat, model)
    return res


def branche_boundaries(F: str, constraint: list):
    '''
    inputs:
        F: function as a string (example: '2*x1 + 6*x2')
        constraint: restrictions in the form of a list of strings (example: ['x1 - x2 <= 4', '-3*x1 <= 17'])
    outputs:
        tuple: (point, result of function);
        example: (array([1., 5.]), 4.00000000000000)
    '''

    exp = sympify(F)

    count_constraint = int(input('Введите количество ограничений: '))

    symbs = sorted(list(exp.free_symbols), key=lambda x: str(x))

    c = [float(exp.coeff(symb)) for symb in symbs]

    dic_constr = {i: [] for i in range(1, count_constraint+1)}
    for i in range(len(constraint)):
        ineq = sympify(constraint[i])
        ex = ineq.args[0]
        dic_constr[i+1] = [float(ex.coeff(symb)) for symb in symbs]

    A = [v for v in dic_constr.values()]

    for i in range(len(symbs)):
        lst = [0.0]*len(symbs)
        lst[i] = -1.0
        A.append(lst)

    b = []
    for ineq in constraint:
        ine = sympify(ineq)
        b.append(ine.args[1])

    b += [0]*len(symbs)

    res = linprog(c, A_ub=A, b_ub=b, method='simplex')['x']

    if all(str(i).split('.')[-1] == '0' for i in res) \
            and all([sympify(const).subs(list(zip(symbs, res))) for const in constraint]):
        return res

    try:
        res_zlp1 = ZLP1(F, constraint, res, symbs, A, b, c)
    except:
        res_zlp1 = None

    try:
        res_zlp2 = ZLP2(F, constraint, res, symbs, A, b, c)
    except:
        res_zlp2 = None

    if res_zlp1 is not None and res_zlp2 is not None:
        if exp.subs(list(zip(symbs, res_zlp1))) <= exp.subs(list(zip(symbs, res_zlp2))):
            return res_zlp1, exp.subs(list(zip(symbs, res_zlp1)))
        else:
            return res_zlp2, exp.subs(list(zip(symbs, res_zlp2)))
    elif res_zlp1 is not None and res_zlp2 is None:
        return res_zlp1, exp.subs(list(zip(symbs, res_zlp1)))
    elif res_zlp1 is None and res_zlp2 is not None:
        return res_zlp2, exp.subs(list(zip(symbs, res_zlp2)))
    else:
        return None


def ZLP1(func, constraint, res_last, symbs, A, b, c):
    i = np.argmax(res_last%1)
    whole = int(res_last[i])
    lst = [0]*len(symbs)
    lst[i] = 1


    A_new = deepcopy(A)
    b_new = deepcopy(b)
    A_new.append(lst)
    b_new.append(whole)

    res = linprog(c, A_ub=A_new, b_ub=b_new, method='simplex')['x']


    if all(str(i).split('.')[-1] == '0' for i in res) \
            and all([sympify(const).subs(list(zip(symbs, res))) for const in constraint]):
        return res

    res_zlp3 = ZLP1(func, constraint, res, symbs, A_new, b_new, c)
    res_zlp4 = ZLP2(func, constraint, res, symbs, A_new, b_new, c)


    to_return = []
    if all(str(i).split('.')[-1] == '0' for i in res_zlp3) \
            and all([sympify(const).subs(list(zip(symbs, res_zlp3))) for const in constraint]):
        to_return.append(res_zlp3)
    if all(str(i).split('.')[-1] == '0' for i in res_zlp4) \
            and all([sympify(const).subs(list(zip(symbs, res_zlp4))) for const in constraint]):
        to_return.append(res_zlp4)

    if to_return:
        return to_return
    else:
        return None


def ZLP2(func, constraint, res_last, symbs, A, b, c):
    i = np.argmax(res_last%1)
    whole = -int(res_last[i])-1 
    lst = [0]*len(symbs)
    lst[i] = 1

    A_new = deepcopy(A)
    b_new = deepcopy(b)
    A_new.append(lst)
    b_new.append(whole)

    res = linprog(c, A_ub=A_new, b_ub=b_new, method='simplex')['x']

    if all(str(i).split('.')[-1] == '0' for i in res) \
            and all([sympify(const).subs(list(zip(symbs, res))) for const in constraint]):
        return res

    res_zlp5 = ZLP1(func, constraint, res, symbs, A_new, b_new, c)
    res_zlp6 = ZLP2(func, constraint, res, symbs, A_new, b_new, c)

    to_return = []
    if all(str(i).split('.')[-1] == '0' for i in res_zlp5) \
            and all([sympify(const).subs(list(zip(symbs, res_zlp5))) for const in constraint]):
        to_return.append(res_zlp5)
    if all(str(i).split('.')[-1] == '0' for i in res_zlp5) \
            and all([sympify(const).subs(list(zip(symbs, res_zlp5))) for const in constraint]):
        to_return.append(res_zlp6)

    if to_return:
        return to_return
    else:
        return None


def T(i): # охлаждение
    return 1 / i


def P(E_new, E_old, T): # критерий мегаполиса
    return np.exp(-1 * (float(E_new) - float(E_old)) / T)


def neighbour(F, symbs, constraints, point, i):
    d = np.random.normal()*T(i)*gradE(F, symbs, point)
    while const(symbs, constraints, point + d) == False:
        d = np.random.normal()*T(i)*gradE(F, symbs, point)
    return d


def const(symbs, constraints, point):
    for const in constraints:
        sp_const = sympify(const)
        if not sp_const.subs(list(zip(symbs, point))):
            return False
    return True


def gradE(F, symbs, point):
    diffs = []
    for symb in symbs:
        diffs.append(F.diff(symb).subs(list(zip(symbs, point))))
    return np.array(diffs)


def simulated_annealing(F: str, constraints: list, start_points: list, steps=1000):
    '''
    inputs:
        F: function as a string (example: '2*x1 + 6*x2')
        constraints: restrictions in the form of a list of strings (example: ['x1 - x2 <= 4', '-3*x1 <= 17'])
        start_points: start point (example: [0, 0])
        steps: count steps of loop
    outputs:
        tuple: point;
        example: [1, 1]
    '''
    exp = sympify(F)
    symbs = sorted(list(exp.free_symbols), key=lambda x: str(x))

    check_constraints = const(symbs, constraints, start_points)
    if not check_constraints:
        return 'Точка не удовлетворяет ограничениям'

    x_old = np.array(start_points)
    for i in range(1, steps+1):
        t = T(i)
        x_new = x_old + neighbour(exp, symbs, constraints, x_old, i)
        E_old = exp.subs(list(zip(symbs, x_old)))
        E_new = exp.subs(list(zip(symbs, x_new)))
        if E_new - E_old < 0:
            x_old = x_new
        else:
            if P(E_new, E_old, t) >= 0.9:
                x_old = x_new

    return x_new


def get_data(func, restr):
    func = sympify(func)
    symbs = list(func.free_symbols)
    number_of_genes = len(symbs)
    c = []
    restrictions = []
    for i in range(len(restr)):
            restrictions.append(sympify(restr[i][:restr[i].index('>')].replace(' ', '')))
            c.append(restr[i][restr[i].index('>')+2:].replace(' ', ''))
    return {'func' : func,
            'restr' : restrictions,
            'number_of_genes': number_of_genes, 
            'c': c, 
            'symbs' : symbs}


def check_restrictions(gene, restr, c, symbs):
    flag = True
    for i in range(len(restr)):
        if sympify(restr[i]).subs(list(zip(symbs, list(gene)))) < float(c[i]):
            flag = False
    return flag


def get_zero_population(number_of_genes, restr, c, symbs):
    len_of_population = 500
    zero_population = []
    while len(zero_population) < len_of_population:
        x = np.random.uniform(-10, 10, number_of_genes)
        if check_restrictions(x, restr, c, symbs) == True:
            zero_population.append(x)
    return zero_population


def evaluate(population, func, symbs):
    finding_best = []
    for gene in population:
        value = func.subs(list(zip(symbs, list(gene))))
        finding_best.append((list(gene), value))
    finding_best.sort(key = lambda x: x[1])
    best = []
    for i in range(len(finding_best)):
        best.append(finding_best[i][0])
    return best


def recombination(parents, number_of_genes):
    positions = []
    for i in range(number_of_genes):
        c = []
        for j in range(len(parents)):
            c.append(parents[j][i])
        positions.append(c)
    children_tuple = list(itertools.product(*positions))
    children = [list(child) for child in children_tuple]
    return children


def get_new_population(evaluated, strength_of_mutation, restr, number_of_genes, c, symbs):
    if len(evaluated) > 10:
        n = int(len(evaluated)/2)
    else:
        n = len(evaluated)
    p = 2
    parents = evaluated[:p]
    children = recombination(parents, number_of_genes)

    new_population = evaluated[:n]
    new_population.extend(children)
    number_of_mutations = random.choice(list(range(len(children))))
    for i in range(number_of_mutations):
        index = random.choice(list(range(number_of_mutations)))
        new_population[index] = [gene*strength_of_mutation for gene in new_population[index]]
    population = []
    for gene in new_population:
        if check_restrictions(list(gene), restr, c, symbs) == True:
            population.append(gene)
    return population


def genetic_algorithm(func, restr, strength_of_mutation=1, number_of_generations=20):
    """
    Solving an optimisation problem for a function with inequality-type
    constraints by Genetic Algorithm.
    Returns the found point.
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
    res = get_data(func, restr)
    func = res['func']
    restr = res['restr']
    number_of_genes = res['number_of_genes']
    c = res['c']
    symbs = res['symbs']

    population = get_zero_population(number_of_genes, restr, c, symbs)

    for generation in range(number_of_generations):
        evaluated = evaluate(population, func, symbs)
        population = get_new_population(evaluated, strength_of_mutation, restr, number_of_genes, c, symbs)

    evaluated = evaluate(population, func, symbs)
    return evaluated[0]
  
  
import numpy as np
import matplotlib.pyplot as plt

def own_pegasos(X_train, y_train, X_test, lam = 0.001, max_iter = 1000, stoch_size = 0.2, draw = False):
    """
    A function that implements a two-class classification model using
    the pegasos method for the reference vector
    method training problem.
    Parameters
    ----------
    X_train : numpy.array
        A set of features for the training sample.
    y_train : numpy.array
        Set of class values for the training sample (classes can be only -1 or 1).
    X_test : numpy.array
        A set of features for the test sample.
    lam: float, default= 0.001
        the power of regularization
    max_iter : int, default=1000
        Number of learning iterations
    stoch_size: float, default = 0.2
        the proportion of the training sample for stochastic gradient descent
    draw : bool, default=False
        Plotting the classification graph.

    Examples
    >>> from sklearn.datasets import make_blobs
    >>> X, Y = make_blobs(n_samples = 40,centers=2, cluster_std=1.2,n_features=2,random_state=42)
    >>> for i,j in enumerate(Y):
    >>>     if j == 0:
    >>>         Y[i] = -1
    >>>     elif j == 1:
    >>>         Y[i] = 1
    >>> X_train = X[:35]
    >>> Y_train = Y[:35]
    >>> X_test= X[35:]
    >>> Y_test = Y[35:]
    >>> own_pegasos(X_train, Y_train, X_test)
    >>> {'weights': array([ 0.38290961, -0.09323558,  0.05267501]),
    >>> 'y_pred': array([ 1, -1,  1, -1,  1])}
    """
    if len(y_train.shape) != 1 or (np.unique(y_train) == np.array([-1, 1])).sum() != 2:
        raise ValueError('������ ������� ������ ���� ���������� � ��������� ������ ��� ������: -1 ��� 1')
    if X_train.shape[1] != X_test.shape[1]:
        raise ValueError('���-�� ��������� � ��������� � �������� ������� ������ ���� ����������')
    k= stoch_size * X_train.shape[0]
    inds =np.arange(X_train.shape[0])
    #���������� �������� �� ������
    X_train = np.concatenate((X_train, np.ones((X_train.shape[0],1))), axis=1)
    X_test = np.concatenate((X_test, np.ones((X_test.shape[0],1))), axis=1)
    w = np.zeros(X_train.shape[1])
    margin_current = 0
    margin_previous = -10
    not_converged = True
    t = 0

    while(not_converged):
        margin_previous = margin_current
        t += 1
        pos_support_vectors = 0
        neg_support_vectors = 0
        eta = 1/(lam*t)
        fac = (1-(eta*lam))*w
        np.random.shuffle(inds)
        selected_inds = inds[:round(k)]
        for i in selected_inds:
            prediction = np.dot(X_train[i], w)

            if (round((prediction),1) == 1):
                pos_support_vectors += 1
            if (round((prediction),1) == -1):
                neg_support_vectors += 1

            if y_train[i]*prediction < 1:
                w = fac + eta*y_train[i]*X_train[i]
            else:
                w = fac

        if t>max_iter:    
            margin_current = np.linalg.norm(w)
            if((pos_support_vectors > 0)and(neg_support_vectors > 0)and((margin_current - margin_previous) < 0.01)):
                not_converged = False

    y_pred = []
    for i in X_test:
        pred = np.dot(w,i)
        if (pred > 0):
            y_pred.append(1)
        elif (pred < 0):
            y_pred.append(-1)

    if X_test.shape[1] == 3 and draw:
        plt.scatter(X_test[:, 0], X_test[:, 1], c= y_pred, s=20, cmap='viridis')
        plt.xlabel('������� 1')
        plt.ylabel('������� 2')
    elif draw:
        raise ValueError('�� ����� ��������� ������')
    return {'y_pred': np.array(y_pred), 'weights': w}
