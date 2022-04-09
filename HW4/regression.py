from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.preprocessing import PolynomialFeatures
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import warnings


class InsufficientData(Exception):
    def __str__(self):
        string_exp = 'Данных недостаточно. Их должно быть не менее 2^k строк, \
где k – количество признаков. Если признак 1, то хотя бы 10 строк.'
        return string_exp


class LinearlyDependent(Exception):
    def __str__(self):
        return 'Присутствуют линейно зависимые признаки. Мы не можем применить МНК.'


class DegreeError(Exception):
    def __str__(self):
        return 'Степень полинома должна быть целым неотрицательным числом.'


class NegativeValue(Exception):
    def __str__(self):
        return 'Значения y должны быть положительными'


class VeryBig(Exception):
    def __str__(self):
        return 'Свободный член получился слишком большим, чтобы произвести вычисления'

    
def student_del(X, y):
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
    if len(X.shape) == 2 and X.shape[1] > 1:
        if X.shape[0] < 2**X.shape[1] or len(X) < 10:
            raise InsufficientData
    else:
        if len(X) < 10:
            raise InsufficientData


def plot_3d_regression(X, y, coef, a0, n_point):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(X.iloc[:, 0], X.iloc[:, 1], y, marker='.', color='red')
    ax.set_xlabel("X0")
    ax.set_ylabel("X1")
    ax.set_zlabel("y")

    xs = np.tile(np.arange(n_point), (n_point,1))
    ys = np.tile(np.arange(n_point), (n_point,1)).T
    zs = a0 * np.exp(coef[0]*xs) * np.exp(ys * coef[1])
    ax.plot_surface(xs, ys, zs, alpha=0.5)
    plt.show()


def plot_2d_regression(X, y, coef, a0, reg_type):
    xs = np.linspace(X.min()-1, X.max()+1)
    if reg_type == 'lin':
        zs = a0 + xs*coef
    else:
        zs = a0 * np.exp(coef*xs)
    plt.plot(xs, zs, color="blue", linewidth=3)
    plt.scatter(X, y, marker='.', color='red') 
    plt.show()



def exp_regression(X, y, tol=5, regularization=None, alpha=1.0, draw=False):
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
        plot_3d_regression(X, y, weights, bias)
    elif draw == True and X.shape[1] == 1:
        plot_2d_regression(X, y, weights, bias, reg_type='exp')
    return {'func': func, 
            'weights': weights, 
            'bias': bias}



def poly_regression(X: pd.DataFrame, y: list, degree, regularization=None, alpha=1.0, draw=False, n_point=7000) -> dict:
    if degree < 0 and int(degree) != float(degree):
        raise DegreeError
     
    X, y_new = student_del(X, y)
   
    if X.shape[0] < 2**X.shape[1]:
        raise InsufficientData
    else:
        X = X.to_numpy().reshape(-1, 1)
           
    if regularization is None:
        p = PolynomialFeatures(degree=degree)
        X = p.fit_transform(X)
        reg = LinearRegression().fit(X, y)
    elif regularization == 'L1':
        p = PolynomialFeatures(degree=degree)
        X = p.fit_transform(X)
        reg = Lasso(alpha=alpha).fit(X, y)
    elif regularization == 'L2':
        p = PolynomialFeatures(degree=degree)
        X = p.fit_transform(X)
        reg = Ridge(alpha=alpha).fit(X, y)
    
    # Стьюдент
    weights, bias = reg.coef_, reg.intercept_
    if X.shape[1] == 2:
        func = f'{round(bias, tol)}'
        k = len(weights)
        for i in range(k):
            coef = weights[i]
            func = func + f' * {round(coef, tol)}*x{k-i}'
    else:
        weights = weights[0]
        bias = bias
        func = f'{round(bias, tol)}' + f' * {round(weights, tol)}*x0'
    if draw == True and X.shape[1] > 2:
        print('К сожалению, мы не можем построить график, так как размерность пространства признаков велика.')
    elif draw == True and X.shape[1] == 2:
        plot_3d_regression(X, y, weights, bias, n_point)
    elif draw == True and X.shape[1] == 1:
        plot_2d_regression(X, y, weights, bias, n_point)
    return {'func': func, 
            'weights': weights, 
            'bias': bias}


def lin_regression(X, y, tol = 5, regularization = None, alpha=1.0, draw = False):
    y_new = y.to_numpy()
    check_data(X)

    if regularization is None:
        reg = LinearRegression().fit(X, y_new)
    elif regularization == 'L1':
        reg = Lasso(alpha=alpha).fit(X, y_new)
    elif regularization == 'L2':
        reg = Ridge(alpha=alpha).fit(X, y_new)
        #требует доработки
    elif regularization == 'Student':
        X_new, y_new = student_del(pd.DataFrame(X), 
                                       pd.DataFrame(y_new))
        check_data(X_new)
        if len(X_new.shape) < 2:
            X_new = X_new.to_numpy().reshape(-1, 1)
        reg = LinearRegression().fit(x_new, y_new)

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
        plot_3d_regression(X, y, weights, bias)
    elif draw == True and X.shape[1] == 1:
        plot_2d_regression(X, y, weights, bias, reg_type='lin')
    return {'func': func[:-1], 
            'weights': weights, 
            'bias': bias}
