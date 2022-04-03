import inspect
import pandas as pd
import numpy as np
from numpy.linalg import norm
from scipy.optimize import line_search

def get_output(func, x_new, dataset_rec, flag,  dataset=None):
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

def GD(func, diff_func, lr_rate=None, x_old=None, accuracy=10**-5, maxiter=500, 
       interim_results=False, dataset_rec=False):
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
        x_new = [x_old[i] - lr_rate * diff_func(*x_old)[i] for i in range(shape_arg)]
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
    

def GDSS(func, diff_func, lr_rate=None, e=0.1, d=0.5, x_old=None, accuracy=10**-5, maxiter=500, 
         interim_results=False, dataset_rec=False):
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
        x_new = [x_old[i] - lr_rate * diff_func(*x_old)[i] for i in range(shape_arg)]
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
