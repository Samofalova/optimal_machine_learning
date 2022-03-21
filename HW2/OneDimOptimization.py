def golden_ratio(func, search_area, extreme_type='min', accuracy=10**(-5),
                 maxiter=500, interim_results=False, dataset_rec=False):
    """
    Returns dict with the minimum or maximum of a function of one variable on 
    the segment [a, b] using the golden ratio method, value of function, report
    and intermediate results in pandas.DataFrame (optional).
    Given a function of one variable and a possible bracketing interval,
    return the minimum or maximum of the function isolated to a fractional
    precision of accuracy.
    Parameters
    ----------
    func : str
        Objective function to minimize or maximize.
    search_area : tuple or list
        [a, b], where (a<b) – the interval within which the maximum
        and minimum are searched.
    accuracy : float, optional
        x tolerance stop criterion.
    maxiter : int
        Maximum number of iterations to perform.
    interim_results : bool, optional
        If True, print intermediate results.
    dataset_rec : bool, optional
        If True, an entry in pandas.DataFrame intermediate results.
    Examples
    --------
    >>> from HW2.OneDimOptimization import golden_ratio
    >>> minimum = golden_ratio(func='x**2', search_area=(1, 2))
    >>> print(minimum['point'])
    1.0000048224378428
    """
    str_error = 'Длина отрезка должна быть больше заданной точности'
    assert search_area[1] - search_area[0] > accuracy, str_error
    res = {'point': None, 'value_func': None, 'report': None,
           'interim_results_dataset': None}
    df = pd.DataFrame(columns=['a', 'b', 'x1', 'x2', 'f1', 'f2'])
    x = sp.Symbol('x')
    try:
        func = eval(func)
    except NameError:
        print('Функция должна быть задана через x')
        return None
    proportion = 1.6180339887
    a, b = search_area
    iterat = 0
    while abs(b - a) >= accuracy and iterat != maxiter:
        iterat += 1
        x1 = b - (b-a) / proportion
        x2 = a + (b-a) / proportion
        f1, f2 = float(func.subs({x: x1})), float(func.subs({x: x2}))
        if dataset_rec:
            d = {'a': a, 'b': b, 'x1': x1, 'x2': x2, 'f1': f1, 'f2': f2}
            df = df.append(d, ignore_index=True)

        if extreme_type == 'min':
            if f1 >= f2:
                a = x1
            else:
                b = x2
        elif extreme_type == 'max':
            if f1 <= f2:
                a = x1
            else:
                b = x2

        if interim_results:
            print(f'''{iterat}:
            a = {a}, b = {b},
            x1 = {x1}, x2 = {x2},
            f1 = {f1}, f2 = {f2}''')

    if dataset_rec:
        res['interim_results_dataset'] = df
    res['point'] = (a+b) / 2
    res['value_func'] = func.subs({x: res['point']})
    if abs(b-a) <= accuracy:
        res['report'] = 0
    elif iter == maxiter:
        res['report'] = 1
    else:
        res['report'] = 2
    return res
  
  

def parabola_method(func: str,
                    limits: list,
                    accuracy: float=10**(-5),
                    max_iterations: int=500,
                    intermediate_results: bool=False,
                    intermediate_writing: bool=False):
    """
    Search for the extremum of a function of one variable using the parabola method.
    
    args:
        mandatory:
            - func - function in analytical form;
            - limits - optimization area boundaries;
        optional:
            - accuracy - optimization precision by argument (default: 10^-5);
            - max_iterations - maximum number of iterations (default: 500;
            - intermediate_results - flag "output intermediate results" (default: False);
            - intermediate_writing - flag "writing intermediate results to dataset" (default: False);
            
    outputs:
        - The found value of the extremum point coordinate;
        - Function value at the extremum point;
        - Algorithm report;
    """
    
    x = symbols('x')
    F = eval(func)
    
    
    left_hand, right_hand = limits
    
    left_hand = float(left_hand)
    right_hand = float(right_hand)
    
    if left_hand > right_hand:
        left_hand, right_hand = right_hand, left_hand
        
    iteration_num = 0
    d = 100
    while (d > accuracy) or (iteration_num < max_iterations):
        # need to satisfy inequality x1<x2<x3, f(x1) >= f(x2) <= f(x3) 

        # first step
        if iteration_num == 0:
            x1 = left_hand
            x3 = right_hand
            x2 = (x1+x3)/2
        

        f1 = F.subs(x, x1)
        f2 = F.subs(x, x2)
        f3 = F.subs(x, x3)
        
        #return f1

        # second step
        
        # first  formula - a0 = f1, a1 = (f2-f1)/(x2-x1), a2 = 1/(x3-x2) * ((f3-f1)/(x3-x1) - (f2-f1)/(x2-x1))
        # second formula - x_ = 1/2*(x1 + x2 - a1/a2)
        a0 = float(deepcopy(f1))
        a1 = float((f2-f1)/(x2-x1))
        a2 = float(1/(x3-x2) * ((f3-f1)/(x3-x1) - (f2-f1)/(x2-x1)))

        
        if iteration_num > 0:
            x_old = deepcopy(x_)
            
        x_ = 1/2*(x1 + x2 - a1/a2)
        
        # check num of step - go to step 4 if its first step
        if iteration_num > 0:
            d = abs(x_old - x_)
        
            if d <= accuracy:
                x_res = deepcopy(x_)


        # step 4
        f_x_ = F.subs(x, x_)

        # step 5
        # suppose x1 = x_ = ..., f1 = f_x_ = ...
        if x1 < x_ < x2 < x3 and f_x_ >= f2: # x* in [x_, x3]
            x1 = deepcopy(x_)
            f1 = deepcopy(f_x_)
        elif x1 < x2 < x_ < x3 and f2 >= f_x_: # x* in [x2, x3]
            x1 = deepcopy(x2)
            f1 = deepcopy(f2)
            x2 = deepcopy(x_)
            f2 = deepcopy(f_x_)

        iteration_num += 1

    f_res = F.subs(x, x_res)

    return f_res, x_res, iteration_num
  
  
