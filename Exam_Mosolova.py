#!/usr/bin/env python
# coding: utf-8

# In[17]:


import numpy as np
from sympy import *
from sympy.core.numbers import NaN
#from .first_phase import first_phase
from copy import deepcopy


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

    #if len(start_point) == 0:
    #    start_point = first_phase(restrictions, symbs)
    #if start_point == 'Введенные ограничения не могут использоваться вместе':
    #    return start_point
    #elif start_point == 'Невозможно подобрать внутреннюю точку для данного метода':
    #    return start_point

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


# In[19]:


f = '-x1+4*x2'
c = ['3*x1-x2+6 >= 0', '-x1-2*x2+4 >= 0', 'x2+3 >= 0']
log_barriers(f, c)


# In[ ]:




