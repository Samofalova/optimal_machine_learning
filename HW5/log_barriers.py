from sympy import *
import numpy as np

def log_barriers(func: str, restrictions: list, start_point: tuple, accuracy:float = 10**(-6), max_steps: int=500):
    tao = 1
    v = 10
    phi = f'{tao}*({func})'
    for exp in restrictions:
        phi += f' - log({exp})'
        
    X = Matrix([sympify(phi)])
    symbs = list(sympify(phi).free_symbols)
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
    
    try:
        res_new = sympify(func).subs(list(zip(symbs, next_point)))
    except:
        return res
        
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
        next_point = [start_point[i]-xk[i] for i in range(len(start_point))]
        try:
            res_new = sympify(func).subs(list(zip(symbs, next_point)))
        except:
            return res_new
        tao = tao*v
        steps += 1
        
    res_new = sympify(func).subs(list(zip(symbs, next_point)))

    return res_new

