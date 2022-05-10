import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import PolynomialFeatures

def plot_boundary(clf, X, y, grid_step=.01, poly_featurizer=None):
    x_min, x_max = X[:, 0].min() - .1, X[:, 0].max() + .1
    y_min, y_max = X[:, 1].min() - .1, X[:, 1].max() + .1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, grid_step),
    np.arange(y_min, y_max, grid_step))
    Z = clf.predict(poly_featurizer.transform(np.c_[xx.ravel(), yy.ravel()]))
    Z = Z.reshape(xx.shape)
    plt.contour(xx, yy, Z, cmap=plt.cm.Paired)


def draw_2d(X_test, y_hat, logit, poly):
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
