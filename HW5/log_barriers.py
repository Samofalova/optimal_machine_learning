import numpy as np
from sympy import *
from sympy.core.numbers import NaN
from .first_phase import first_phase
from copy import deepcopy


def log_barriers(func: str, restrictions: list, start_point: tuple = tuple(), accuracy:float = 10**(-6), max_steps: int=500):
    tao = 1
    v = 10
    for i in range(len(restrictions)):
        restrictions[i] = restrictions[i][:restrictions[i].index('>')].replace(' ', '')
        
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
        return res
        
    steps = 1
    return res_new
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
        res_new = sympify(func).subs(list(zip(symbs, next_point)))
        if type(res_new) == NaN:
            return old_point

        tao = tao*v
        steps += 1

    return next_point

