import numpy as np
from sympy import *
from scipy.optimize import linprog

def branch_boundaries(F: str, constraint: list):
    
    exp = sympify(F)
    
    count_constraint = int(input('Введите количество ограничений: '))
    
    symbs = sorted(list(exp.free_symbols), key=lambda x: str(x))
    
    c = [float(exp.coeff(symb)) for symb in symbs]
    
    dic_constr = {i: [] for i in range(1, count_constraint+1)}
    for i in range(len(constraint)):
        ineq = sympify(constraint[i])
        print(ineq)
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
        b.append(float(ine.args[1]))
    
    b += [0]*len(symbs)
    
    res = linprog(c, A_ub=A, b_ub=b, method='simplex')['x']
    
    if str(sum(res))[-1] == '0':
        return res
    else:
        # ЗЛП-1
        i = np.argmax(res%1)
        whole = int(res[i])
        lst = [0]*len(symbs)
        lst[i] = 1
        A.append(lst)
        b.append(whole)
        
        # ЗЛП-2
        i = np.argmax(res%1)
        whole = -int(res[i])-1
        ...
    
