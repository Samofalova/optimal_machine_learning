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
                 n, m, p, Fi, func, tol):
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
    us : tuple
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
                                n, m, p, Fi, func, tol)
        return point, y
    except:
        str1 = 'Мы сделали все возможное, чтобы решить эту задачу. '
        str2 = 'К сожалению, не помогла даже молитва((('
        print(str1+str2)
        raise


