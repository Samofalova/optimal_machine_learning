import sympy as sp
import scipy.linalg as la
import numpy as np
import matplotlib.pyplot as plt
from sympy import I, sin, cos, log, pi, tan, sqrt, exp
from functools import partial
from sympy.plotting.plot import MatplotlibBackend, Plot
from sympy.plotting import plot3d
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
    data['X'] = sp.symbols(input(text).split())
    assert len(data['X']) == 2, 'переменные заданы неверно'

    f = input('Введите функцию (y*x+2): ')
    data['func'] = sp.Matrix([f])

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


def get_crit(func: sp.Matrix, X: list):
    """
    func: sympy.Matrix(['x + y']),
    X: [sympy.Symbol('x'), sympy.Symbol('y')]
    return critical points
    """
    gradf = sp.simplify(func.jacobian(X))
    return sp.solve(gradf, X, dict=True)


def filter_point(point: list, x_min, x_max, y_min, y_max):
    """
    point: [(1, 2), (2, 3)] – list of tuples, critical points for filtering
    x_min, x_max, y_min, y_max – int or float, constraints for variables
    """
    x, y = point.values()
    if sp.simplify(x).is_real and sp.simplify(y).is_real:
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
    hessianf = sp.simplify(sp.hessian(func, X))
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
                x3.append(data['func'].subs(dict(zip(data['X'], point[0]))))
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
