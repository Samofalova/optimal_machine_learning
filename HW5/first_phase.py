from sympy import *
import numpy as np

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

