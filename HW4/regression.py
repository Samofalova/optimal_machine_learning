from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.preprocessing import PolynomialFeatures
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


class InsufficientData(Exception):
    def __str__(self):
        return 'Данных недостаточно. Их должно быть не менее 2^k, где k – количество признаков.'


class LinearlyDependent(Exception):
    def __str__(self):
        return 'Присутствуют линейно зависимые признаки. Мы не можем применить МНК.'

class DegreeError(Exception):
    def __str__(self):
        return 'Степень полинома должна быть целым неотрицательным числом.'


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


def plot_2d_regression(X, y, coef, a0, n_point): 
    xs = np.arange(n_point)
    zs = a0 * np.exp(coef*xs)
    plt.plot(xs, zs, color="blue", linewidth=3)
    plt.scatter(X, y, marker='.', color='red')
    plt.show()


def exp_regression(X, y, tol=5, regularization=None, alpha=1.0, draw=False, n_point=7000):
    y_new = np.log(y)
    if len(X.shape) == 2:
        if X.shape[0] < 2**X.shape[1]:
            raise InsufficientData
    else:
        X = X.to_numpy().reshape(-1, 1)
            
    if regularization is None:
        if X.shape[1] >= 2 and np.linalg.det(X.T@X) == 0:
            raise LinearlyDependent
        reg = LinearRegression().fit(X, y_new)
    elif regularization == 'L1':
        reg = Lasso(alpha=alpha).fit(X, y_new)
    elif regularization == 'L2':
        reg = Ridge(alpha=alpha).fit(X, y_new)
    # Стьюдент
    weights, bias = reg.coef_, np.exp(reg.intercept_)
    if X.shape[1] == 2:
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
        plot_3d_regression(X, y, weights, bias, n_point)
    elif draw == True and X.shape[1] == 1:
        plot_2d_regression(X, y, weights, bias, n_point)
    return {'func': func, 
            'weights': weights, 
            'bias': bias}


def poly_regression(X: pd.DataFrame, y: list, degree, regularization=None, alpha=1.0, draw=False, n_point=7000) -> dict:
    if degree < 0 and int(degree) != float(degree):
        raise DegreeError
    
    if X.shape[0] < 2**X.shape[1]:
        raise InsufficientData
    else:
        X = X.to_numpy().reshape(-1, 1)
        
    if regularization is None:
        reg = PolynomialFeatures(degree=degree).fit(X, y)
    elif regularization == 'L1':
        reg = Lasso(alpha=alpha).fit(X, y)
    elif regularization == 'L2':
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
