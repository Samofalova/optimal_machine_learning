#!/usr/bin/env python
# coding: utf-8

# In[131]:


import numpy as np
from sympy import *
from sympy.core.numbers import NaN
from copy import deepcopy


def log_barriers(func: str, restrictions: list, start_point: tuple = tuple(), accuracy:float = 10**(-6), max_steps: int=500, tao = 1/1000):
    '''
    Solving an optimisation problem for a function with equality-type
    constraints by log barriers method (a way of solving the dual problem).
    Returns the found point and the value of the function in it.
    Parameters
    ----------
    func : string
        Function for optimisation.
    restrictions : list
        List of strings with given linear restrictions.
    start_point : tuple
        Starting point.
    accuracy : float, default=10**(-6)
        Tolerance for termination.
    max_steps : int, default=500
        Maximum of iterations/
    tao : default = 1/1000
        Parameter.
    '''
    v = 2
    
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

        lst_for_subs =  list(zip(symbs, next_point))
        dfx0 = df.subs(lst_for_subs)
        ddfx0 = ddf.subs(lst_for_subs)

        xk = ddfx0.inv() @ dfx0
        old_point = deepcopy(next_point)
        next_point = [next_point[i]+xk[i] for i in range(len(next_point))]
        res = deepcopy(res_new)
        res_new = sympify(func).subs(list(zip(symbs, next_point)))

        if type(res_new) == NaN:
            return np.array(old_point), res

        tao = tao*v
        steps += 1
        

    return {'x': np.array(next_point), 'y':res_new}


# In[132]:


f = '-x1+4*x2'
c = ['3*x1-x2+6 >= 0', '-x1-2*x2+4 >= 0', 'x2+3 >= 0']
res = log_barriers(f, c, start_point=(1, 1), tao = 1/1000)
res['x']


# In[133]:


res['y']


# In[140]:


get_ipython().system('pip install plotly')


# In[145]:


import matplotlib.pyplot as plt
import plotly.graph_objects as go
x_points = np.linspace(-5, 5)
y_points = np.linspace(-5, 5)
X, Y = np.meshgrid(x_points, y_points)
f = lambda x,y: -x + 4*y
Z = f(X, Y)
scatter = go.Scatter3d(x=[float(res['x'][0])], y=[float(res['x'][1])], z=[float(res['y'])], marker={'size':3})
surface = go.Surface(z=Z, x=X, y=Y, colorscale='RdBu', opacity=0.3,  showscale=True)
fig = go.Figure(data=[scatter, surface])
fig.show()


# In[ ]:




