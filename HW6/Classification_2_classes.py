import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.svm import LinearSVC, SVC


def plot_boundary(clf, X, y, grid_step=.01, poly_featurizer=None):
    """
    Plot boundary for draw_2d.
    """
    x_min, x_max = X[:, 0].min() - .1, X[:, 0].max() + .1
    y_min, y_max = X[:, 1].min() - .1, X[:, 1].max() + .1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, grid_step),
    np.arange(y_min, y_max, grid_step))
    Z = clf.predict(poly_featurizer.transform(np.c_[xx.ravel(), yy.ravel()]))
    Z = Z.reshape(xx.shape)
    plt.contour(xx, yy, Z, cmap=plt.cm.Paired)


def draw_2d(X_test, y_hat, logit, poly):
    """
    Plot 2D-graph.
    """
    colors = ['#' + ''.join([random.choice('ABCDEF0123456789') for i in range(6)])
              for col in range(len(np.unique(y_hat)))]
    for clss in np.unique(y_hat):
        plt.scatter(X_test[y_hat == clss, 0], X_test[y_hat == clss, 1], 
                    c=colors[clss], label=clss)
    plt.xlabel("Признак 1")
    plt.ylabel("Признак 2")
    plot_boundary(logit, X_test, y_hat, grid_step=.01, poly_featurizer=poly)
    plt.legend()


def own_logistic_regression(X_train, y_train, X_test,
                            penalty='none', C=1, l1_ratio='none',
                            degree: int=1, draw=False, max_iter=100):
    """
    A function that implements a two-class classification model based on logistic regression.
    Parameters
    ----------
    X_train : numpy.array
        A set of features for the training sample.
    y_train : numpy.array
        Set of class values for the training sample.
    X_test : numpy.array
        A set of features for the test sample.
    penalty : string, default='none'
        Type of regalarisation. ['l1', 'l2', 'elasticnet', 'none']
    C : float, default=1
        Regalarisation strength.
    l1_ratio : float, default='none'
        Regalarisation strength l1-regalarisation if penalty='elfsticnet' is selected.
    degree : int, default=1
        Polynomial degree.
    draw : int, default=False
        Plotting the classification graph.
    max_iter : int, default=100
        Number of learning iterations
    Examples
    --------
    >>> from HW6.Classification_2_classes.py import *
    >>> Xn, Yn = make_blobs(n_samples=50, centers=2, n_features=2, random_state=1, cluster_std=3)
    >>> X_trainn, X_testn, y_trainn, y_testn = train_test_split(Xn, Yn, test_size=0.2)
    >>> own_logistic_regression(X_trainn, y_trainn, X_testn)
    {'X_test': array([[ -1.08681345,  10.70725528],
        [ -9.5176013 ,  -1.32484178],
        [-14.33005392,  -5.46674614],
        [ -0.70244262,   3.65837874],
        [ -9.40281334,  -3.59632261],
        [ -3.44098628,  -8.14283755],
        [ -3.72107801,   1.87087294],
        [ -7.48076226,  -1.1600423 ],
        [ -2.02823058,   1.59918157],
        [ -2.17684453,   1.77291462]]),
     'y_hat': array([0, 1, 1, 0, 1, 1, 0, 1, 0, 0]),
     'intercept': array([-1.83565162]),
     'coef': array([[-0.62953643, -0.91903067]])}
    """
    param = {'penalty': penalty, 
             'C': C, 
             'l1_ratio': l1_ratio}
    str1 = 'Не выполняется условие 0 <= l1_ratio <= 1'
    str2 = 'Не выполняется условие C > 0'
    str3 = 'Всего один класс, мы не можем обучить модель'
    assert l1_ratio=='none' or 0 <= l1_ratio <= 1, str1
    assert C > 0, str2
    assert len(np.unique(y_train)) > 1, str3
    poly = PolynomialFeatures(degree, include_bias=False)
    X_poly = poly.fit_transform(X_train)
    X_test_poly = poly.transform(X_test)
    if X_poly.shape[0] < 2**X_poly.shape[1]:
        print('Данных недостаточно для обучения коэффициентов. Продолжить?')
        answ = input('0 – Нет\n1 – Да')
        if answ == 0:
            return None
    if penalty == 'elasticnet' and l1_ratio == 'none':
        l1_ratio = 0.8
    if penalty == 'none':
        del param['l1_ratio']
        del param['C']
    if penalty not in ['elasticnet', 'none']:
        del param['l1_ratio']
    model = LogisticRegression(max_iter=max_iter,
                               solver='saga',
                               **param)
    model.fit(X_poly, y_train)
    y_hat = model.predict(X_test_poly)
    res = {'X_test': X_test_poly,
           'y_hat': y_hat,
           'intercept': model.intercept_,
           'coef': model.coef_}
    if X_poly.shape[1] == 2 and draw:
        draw_2d(X_test_poly, y_hat, model, poly)
    return res


def draw_rdf(X, y, clf):
    """
    Plot 2D-graph for SVM. 
    """
    colors = ['#' + ''.join([random.choice('ABCDEF0123456789') for i in range(6)])
              for col in range(len(np.unique(y)))]
    for clss in np.unique(y):
        plt.scatter(X[y == clss, 0], X[y == clss, 1], 
                    c=colors[clss], label=clss)
    plt.xlabel("Признак 1")
    plt.ylabel("Признак 2")
    h = .02
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contour(xx, yy, Z, cmap=plt.cm.Paired)


def logistic_regression_with_rbf(X_train, y_train, X_test,
                                 C=1, gamma='scale',
                                 draw=False, max_iter=100):
    """
    A function implementing a two-class classification model based on logistic
    regression with radial basis functions.
    Parameters
    ----------
    X_train : numpy.array
        A set of features for the training sample.
    y_train : numpy.array
        Set of class values for the training sample.
    X_test : numpy.array
        A set of features for the test sample.
    C : float, default=1
        Regalarisation strength.
    gamma : float, default='scale'
        Kernel factor.
    draw : int, default=False
        Plotting the classification graph.
    max_iter : int, default=100
        Number of learning iterations
    Examples
    --------
    >>> from HW6.Classification_2_classes.py import *
    >>> Xn, Yn = make_blobs(n_samples=50, centers=2, n_features=2, random_state=1, cluster_std=3)
    >>> X_trainn, X_testn, y_trainn, y_testn = train_test_split(Xn, Yn, test_size=0.2)
    >>> logistic_regression_with_rbf(X_trainn, y_trainn, X_testn,
                                     C=1, gamma='scale',
                                     draw=True, max_iter=100)
    {'X_test': array([[-12.26090633,  -0.19474408],
        [ -3.72107801,   1.87087294],
        [ -9.40281334,  -3.59632261],
        [ -1.53291867,   6.15493551],
        [ -0.75904895,   3.34974033],
        [ -3.44098628,  -8.14283755],
        [ -0.70244262,   3.65837874],
        [ -2.23506656,   1.74360298],
        [ -9.14095053,  -1.29792505],
        [  2.72676391,  -1.77393226]]),
    'y_hat': array([1, 0, 1, 0, 0, 1, 0, 0, 1, 0]),
    'intercept': array([-0.12607821]),
    'coef': array([[-0.32377093, -0.89114665, -0.2455269 , -1., -0.84723235,
                    0.52369094,  1.,  0.51354647,  1.,  0.27043942]])}
    """
    str1 = 'Не выполняется условие gamma > 0'
    str2 = 'Не выполняется условие C > 0'
    str3 = 'Всего один класс, мы не можем обучить модель'
    assert gamma == 'scale' or gamma > 0, str1
    assert C > 0, str2
    assert len(np.unique(y_train)) > 1, str3
    model = SVC(kernel='rbf',
                max_iter=max_iter,
                C=C,
                gamma=gamma)
    model.fit(X_train, y_train)
    y_hat = model.predict(X_test)
    res = {'X_test': X_test,
           'y_hat': y_hat,
           'intercept': model.intercept_,
           'coef': model.dual_coef_}
    if X_test.shape[1] == 2 and draw:
        draw_rdf(X_test, y_hat, model)
    return res


def svm_with_pddipm(X_train, y_train, X_test,
                    penalty='none', C=1, max_iter=100,
                    draw=False):
    """
    A function that implements a two-class classification model using
    the direct-double internal point method for the reference vector
    method training problem.
    Parameters
    ----------
    X_train : numpy.array
        A set of features for the training sample.
    y_train : numpy.array
        Set of class values for the training sample.
    X_test : numpy.array
        A set of features for the test sample.
    penalty : string, default='none'
        Type of regalarisation. ['l2', 'none']
    C : float, default=1
        Regalarisation strength.
    draw : int, default=False
        Plotting the classification graph.
    max_iter : int, default=100
        Number of learning iterations
    Examples
    --------
    >>> from HW6.Classification_2_classes.py import *
    >>> Xn, Yn = make_blobs(n_samples=50, centers=2, n_features=2, random_state=1, cluster_std=3)
    >>> X_trainn, X_testn, y_trainn, y_testn = train_test_split(Xn, Yn, test_size=0.2)
    >>> svm_with_pddipm(X_trainn, y_trainn, X_testn, max_iter=10500)
    {'X_test': array([[ -9.40281334,  -3.59632261],
        [-10.6243952 ,  -2.19347897],
        [  3.31984663,   6.63262235],
        [-12.00969936,  -2.82065719],
        [  3.57487539,   2.12286917],
        [ -0.70244262,   3.65837874],
        [ -3.72107801,   1.87087294],
        [ -8.45892304,  -4.84762705],
        [ -2.17684453,   1.77291462],
        [ -1.29908305,   6.2580992 ]]),
    'y_hat': array([1, 1, 0, 1, 0, 0, 0, 1, 0, 0]),
    'intercept': array([-1.16510368]),
    'coef': array([[-0.3087416 , -0.28134706]])}
    """
    param = {'penalty': 'l2', 
             'C': C, 
             'max_iter': max_iter}
    str1 = 'penalty может быть только l2'
    str2 = 'Не выполняется условие C > 0'
    str3 = 'Всего один класс, мы не можем обучить модель'
    assert penalty in ['none', 'l2'], str1
    assert C > 0, str2
    assert len(np.unique(y_train)) > 1, str3
    model = LinearSVC(dual=True, **param)
    model.fit(X_train, y_train)
    y_hat = model.predict(X_test)
    res = {'X_test': X_test,
           'y_hat': y_hat,
           'intercept': model.intercept_,
           'coef': model.coef_}
    if X_test.shape[1] == 2 and draw:
        draw_rdf(X_test, y_hat, model)
    return res
