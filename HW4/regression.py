from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.preprocessing import PolynomialFeatures
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import warnings


class InsufficientData(Exception):
    """
    Rises when there is not enough data to build a model.
    """
    def __str__(self):
        string_exp = 'Данных недостаточно. Их должно быть не менее 2^k строк, \
где k – количество признаков. Если признак 1, то хотя бы 10 строк.'
        return string_exp


class LinearlyDependent(Exception):
    """
    Rises when there is a linear relationship between the signs, which makes it impossible
    to apply the method of least squares.
    """
    def __str__(self):
        return 'Присутствуют линейно зависимые признаки. Мы не можем применить МНК.'


class DegreeError(Exception):
    """
    Rises when an incorrect degree is entered to construct a polynomial regression.
    """
    def __str__(self):
        return 'Степень полинома должна быть целым неотрицательным числом.'


class NegativeValue(Exception):
    """
    Rises when negative values of y have been fed to the input for the exponential regression. 
    """
    def __str__(self):
        return 'Значения y должны быть положительными'


class VeryBig(Exception):
    """
    Rises when the free term in the exponential regression is too large and further calculations 
    are impossible. 
    """
    def __str__(self):
        return 'Свободный член получился слишком большим, чтобы произвести вычисления'
    
class RegularizationError(Exception):
    """
    It raises when one tries to apply regularization to polynomial regression. 
    """
    def __str__(self):
        return 'К сожалению, мы не можем построить полиномиальную регрессию с регулязацией'

    
def student_del(X, y):
    """
    Excludes points from the data according to Student's regularization.
    """
    X_new = X.copy()
    y_new = y.copy()
    for line in range(len(X)):
        if len(X.shape) == 2:
            if X_new.drop(index=line).var().sum() < X_new.var().sum():
                X_new = X_new.drop(index=line)
                y_new = y_new.drop(index=line)
        else:
            if X_new.drop(index=line).var() < X_new.var():
                X_new = X_new.drop(index=line)
                y_new = y_new.drop(index=line)
    return X_new, y_new


def check_data(X):
    """
    Checks the data size for sufficiency to build an adequate model.
    """
    if len(X.shape) == 2 and X.shape[1] > 1:
        if X.shape[0] < 2**X.shape[1] or len(X) < 10:
            raise InsufficientData
    else:
        if len(X) < 10:
            raise InsufficientData


def plot_3d_regression(X, y, coef, a0, reg_type):
    """
    Plots the graph of the function and plots points from the dataset. 
    Works if there are two feature. 
    Parameters
    ----------
    X : pandas.DataFrame
        A dataset of features.
    y : numpy.array or pandas.DataFrame
        A dataset of targets.
    coef : numpy.array
        The coefficients at the features in the resulting function.
    a0: float or numpy.float64
        The free term in the resulting function.
    reg_type : string
        Type of function.
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel("X0")
    ax.set_ylabel("X1")
    ax.set_zlabel("y")
    
    if reg_type=='lin':
        ax.scatter(X.iloc[:, 0], X.iloc[:, 1], y, marker='.', color='red') 

        a = np.arange(min(X.min())-1, max(X.max())+1)
        xs = np.tile(a,(len(a),1))
        ys = np.tile(a, (len(a),1)).T       
        zs = a0 + coef[0]*xs + ys * coef[1]

    elif reg_type=='exp':
        ax.scatter(X.iloc[:, 0], X.iloc[:, 1], y, marker='.', color='red') 

        a = np.arange(min(X.min())-1, max(X.max())+1)
        xs = np.tile(a,(len(a),1))
        ys = np.tile(a, (len(a),1)).T  
        zs = a0 * np.exp(coef[0]*xs) * np.exp(ys * coef[1])
    elif reg_type=='poly1': 
        ax.scatter(X.iloc[:, 1], X.iloc[:, 2], y, marker='.', color='red') 
        a = np.arange(min(X.iloc[:, 1:].min())-1, max(X.iloc[:, 1:].max())+1)
        xs = np.tile(a,(len(a),1))
        ys = np.tile(a, (len(a),1)).T 
        zs = a0 + coef[0]*xs + ys * coef[1]
    else:
        ax.scatter(X.iloc[:, 1], X.iloc[:, 2], y, marker='.', color='red') 
        a = np.arange(X.iloc[:, 1].min(), X.iloc[:, 1].max())
        xs = np.tile(a,(len(a),1))
        ys = np.square(np.tile(a, (len(a),1))).T 
        zs = a0 + coef[0]*xs + xs**2 * coef[1]      

    ax.plot_surface(xs, ys, zs, alpha=0.5)
    plt.show()


def plot_2d_regression(X, y, coef, a0, reg_type):
    """
    Plots the graph of the function and plots points from the dataset. 
    Works if there are one feature. 
    Parameters
    ----------
    X : pandas.DataFrame
        A dataset of features.
    y : numpy.array or pandas.DataFrame
        A dataset of targets.
    coef : numpy.array
        The coefficients at the features in the resulting function.
    a0: float or numpy.float64
        The free term in the resulting function.
    reg_type : string
        Type of function.
    """
    xs = np.linspace(X.min()-1, X.max()+1)
    if reg_type == 'lin':
        zs = a0 + xs*coef
    else:
        zs = a0 * np.exp(coef*xs)
    plt.plot(xs, zs, color="blue", linewidth=3, label='Прогноз')
    plt.scatter(X, y, marker='.', color='red', label='Исходные') 
    plt.legend()
    plt.show()



def exp_regression(X, y, tol=5, regularization=None, alpha=1.0, draw=False):
    """
    Ordinary least squares exponential regression. Fits the model to minimize the residual 
    sum of squares between the observed targets of the data set and the targets predicted 
    by the approximation.
    Parameters
    ----------
    X : pandas.DataFrame
        A dataset of features.
    y : pandas.DataFrame
        A dataset of targets.
    tol : int, default=
        The number of decimal places to round the coefficient when writing the function 
        in analytic form.
    regularization: string, optional
        Type of regularization.
    alpha : float, default=1.0
        Constant for regularization.
    draw : bool, optional
        Flag for the chart. If the value is True, the graph is drawn. Works only for 
        two- and three-dimensional cases.
    Examples
    --------
    >>> from HW4.regression import *
    >>> import pandas as pd
    >>> import yfinance as yf
    >>> aapl = yf.download('AAPL', '2021-01-01', '2022-01-01')
    >>> aapl = aapl.reset_index(level=0)
    >>> exp_regression(aapl['Volume'], aapl['Close'])
    {'func': '143.6968 * exp(-0.0*x0)',
     'weights': -2.697485449323164e-10,
     'bias': 143.6967960345703}
    """
    if not (y > 0).all():
        raise NegativeValue
    y_new = np.log(y)
    check_data(X)
    if len(X.shape) < 2:
        X = X.to_numpy().reshape(-1, 1)
            
    if regularization is None:
        if X.shape[1] >= 2 and np.linalg.det(X.T@X) == 0:
            raise LinearlyDependent
        reg = LinearRegression().fit(X, y_new)
    elif regularization == 'L1':
        reg = Lasso(alpha=alpha).fit(X, y_new)
    elif regularization == 'L2':
        reg = Ridge(alpha=alpha).fit(X, y_new)
    elif regularization == 'Student':
        X_new, y_log_new = student_del(pd.DataFrame(X), 
                                       pd.DataFrame(y_new))
        check_data(X_new)
        if len(X_new.shape) < 2:
            X_new = X_new.to_numpy().reshape(-1, 1)
        reg = LinearRegression().fit(X_new, y_log_new)
    X = pd.DataFrame(X)
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings('error')
            if regularization == 'Student':
                weights, bias = reg.coef_[0], np.exp(reg.intercept_)[0]
            else:
                weights, bias = reg.coef_, np.exp(reg.intercept_)
    except RuntimeWarning:
        raise VeryBig
    if X.shape[1] >= 2:
        func = f'{round(bias, tol)}'
        k = len(weights)
        for i in range(k):
            coef = weights[i]
            func = func + f' * exp({round(coef, tol)}*x{k-i})'
    else:
        weights = weights[0]
        bias = bias
        func = f'{round(bias, tol)}' + f' * exp({round(weights, tol)}*x0)'
    if draw == True and X.shape[1] > 2:
        print('К сожалению, мы не можем построить график, так как размерность пространства признаков велика.')
    elif draw == True and X.shape[1] == 2:
        plot_3d_regression(X, y, weights, bias, reg_type='exp')
    elif draw == True and X.shape[1] == 1:
        plot_2d_regression(X, y, weights, bias, reg_type='exp')
    return {'func': func, 
            'weights': weights, 
            'bias': bias}



def poly_regression(X, y, degree, tol=5, regularization=None, alpha=1.0, draw=False):
    """
    Ordinary least squares polynomial regression. Fits the model to minimize the residual 
    sum of squares between the observed targets of the data set and the targets predicted 
    by the approximation.
    Parameters
    ----------
    X : pandas.DataFrame
        A dataset of features.
    y : pandas.DataFrame
        A dataset of targets.
    tol : int, default=
        The number of decimal places to round the coefficient when writing the function 
        in analytic form.
    regularization: string, optional
        Type of regularization.
    alpha : float, default=1.0
        Constant for regularization.
    draw : bool, optional
        Flag for the chart. If the value is True, the graph is drawn. Works only for 
        two- and three-dimensional cases.
    Examples
    --------
    >>> from HW4.regression import *
    >>> import pandas as pd
    >>> import yfinance as yf
    >>> aapl = yf.download('AAPL', '2021-01-01', '2022-01-01')
    >>> aapl = aapl.reset_index(level=0)
    >>> poly_regression(aapl['Volume'], aapl['Close'], degree=1)
    {'func': '143.4969 + -0.0*x1',
     'weights': -2.805183057424643e-08,
     'bias': 143.49689552425573}
    """
    
    if degree <= 0 or type(degree)!=int: 
        raise DegreeError
        
    check_data(X)
    if len(X.shape) < 2:
        X = X.to_numpy().reshape(-1, 1)
    poly = PolynomialFeatures(degree)
    X_poly = poly.fit_transform(X)
    if regularization is not None and degree>1:
        raise RegularizationError
    
    if X.shape[1] >= 2 and np.linalg.det(X.T@X) == 0:
        raise LinearlyDependent
    reg = LinearRegression().fit(X_poly, y)

    
    if regularization == 'Student':
        weights, bias = reg.coef_[0], reg.intercept_[0]
    else:
        weights, bias = reg.coef_, reg.intercept_    

    X_poly = pd.DataFrame(X_poly)
    
    weights = weights[1:]
    
    if X.shape[1]==1 and degree==1 : 
        weights = weights[0] 
        func = f'{round(bias, tol)} + {round(weights, tol)}*x1'
    elif X.shape[1]==1 and degree>1: 
        func = f'{round(bias, tol)}' 
        for i in range(1, degree+1):
            func = func + f' + {round(weights[i-1], tol)}*x1^{i}'
    elif X.shape[1]>=2 and degree==1:
        func = f'{round(bias, tol)}' 
        for i in range(len(weights)):
            func = func + f' + {round(weights[i], tol)}*x{i+1}'
    else: 
        func = 'К сожалению, мы не можем вывести функцию для множественной полиномиальной регрессии'
    
    if draw == True and X.shape[1] == 1 and degree==1: 
        plot_2d_regression(X, y, weights, bias, reg_type='lin')
    elif draw==True and (X.shape[1]==2 and degree==1):    
        plot_3d_regression(X_poly, y, weights, bias, reg_type='poly1')
    elif draw==True and (X.shape[1]==1 and degree==2):
        plot_3d_regression(X_poly, y, weights, bias, reg_type='poly2')
    else:
        print('К сожалению, мы не можем построить график, так как размерность пространства признаков велика.')
        
    return {'func': func, 
            'weights': weights, 
            'bias': bias}


def lin_regression(X, y, tol = 5, regularization = None, alpha=1.0, draw = False):
    """
    Ordinary least squares linear regression. Fits the model to minimize the residual 
    sum of squares between the observed targets of the data set and the targets predicted 
    by the approximation.
    Parameters
    ----------
    X : pandas.DataFrame
        A dataset of features.
    y : pandas.DataFrame
        A dataset of targets.
    tol : int, default=
        The number of decimal places to round the coefficient when writing the function 
        in analytic form.
    regularization: string, optional
        Type of regularization.
    alpha : float, default=1.0
        Constant for regularization.
    draw : bool, optional
        Flag for the chart. If the value is True, the graph is drawn. Works only for 
        two- and three-dimensional cases.
    Examples
    --------
    >>> from HW4.regression import *
    >>> import pandas as pd
    >>> import yfinance as yf
    >>> aapl = yf.download('AAPL', '2021-01-01', '2022-01-01')
    >>> aapl = aapl.reset_index(level=0)
    >>> lin_regression(aapl[['Open', 'Volume']], aapl['Close'], regularization='L2')
    {'func': '0.33756 + 1.00283x1 -0.0x2',
     'weights': array([ 1.00283393e+00, -6.79315538e-09]),
     'bias': 0.33756283615903726}
    """
    y_new = y.to_numpy()
    check_data(X)

    if len(X.shape) < 2:
        X = X.to_numpy().reshape(-1, 1)

    if regularization is None:
        reg = LinearRegression().fit(X, y_new)
    elif regularization == 'L1':
        reg = Lasso(alpha=alpha).fit(X, y_new)
    elif regularization == 'L2':
        reg = Ridge(alpha=alpha).fit(X, y_new)

    elif regularization == 'Student':
        X_new, y_new = student_del(pd.DataFrame(X), 
                                       pd.DataFrame(y_new))
        check_data(X_new)
        if len(X_new.shape) < 2:
            X_new = X_new.to_numpy().reshape(-1, 1)
        reg = LinearRegression().fit(X_new, y_new)

    if regularization == 'Student':
        weights, bias = reg.coef_[0], reg.intercept_[0]
    else:
        weights, bias = reg.coef_, reg.intercept_
    func = str(round(bias, tol)) + ' '
    for i in range(len(weights)):
        if str(weights[i])[0] == '-':
            func += str(round(weights[i], tol)) + 'x' + str(i + 1) + ' '
        else:
            func += '+ ' + str(round(weights[i], tol)) + 'x' + str(i + 1) + ' '
    if draw == True and X.shape[1] > 2:
        print('К сожалению, мы не можем построить график, так как размерность пространства признаков велика.')
    elif draw == True and X.shape[1] == 2:
        plot_3d_regression(X, y, weights, bias, reg_type='lin')
    elif draw == True and X.shape[1] == 1:
        plot_2d_regression(X, y, weights, bias, reg_type='lin')
    return {'func': func[:-1], 
            'weights': weights, 
            'bias': bias}
