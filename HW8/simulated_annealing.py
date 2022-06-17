import numpy as np
import pandas as pd
from sympy import *
from copy import deepcopy

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
