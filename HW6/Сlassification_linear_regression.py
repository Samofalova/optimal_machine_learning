import numpy as np
from matplotlib import pyplot
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

def logistic_regression_with_l1(x_train: np.array, y_train: np.array, x_test: np.array,
        penalty=None, C=1, l1_ratio=None, draw=False, degree=1):
    
    # проверка параметров
    assert C > 0, 'Сила регуляризации дожна быть больше, чем 0!'
    if l1_ratio is not None:
        assert 0 <= l1_ratio <= 1, 'Параметр l1_ration должен быть удовлетворять неравенству 0 <= l1_ratio <= 1!'
    assert degree > 0 and isinstance(degree, int), 'Параметр degree должен быть целым положительным числом!'
    assert len(np.unique(y_train)), 'В y_train должно быть больше одного класса!'
    
    if x_train.shape[0] < 2**x_train.shape[1]:
        print('Для оптимального результата количество наблюдений должно быть больше 2^k.')
        IsContinue = int(input('Все равно продолжить? (0/1): '))
        if not IsContinue:
            return
    
    poly = PolynomialFeatures(degree, include_bias=False)
    
    x_poly = poly.fit_transform(x_train)
    x_poly_test = poly.transform(x_test)
    
    if penalty == 'l1':
        penalty = 'elasticnet'
        l1_ratio = 1
    elif penalty != 'l1':
        penalty = 'elasticnet'
        if l1_ratio is None:
            l1_ratio = 0.8
    
    model = LogisticRegression(max_iter=50_000, penalty=penalty, solver='saga', l1_ratio=l1_ratio, C=C)
    model.fit(x_poly, y_train)
    y_pred = model.predict(x_poly_test)
    
    res = {'x_test': x_poly_test,
           'y_pred': y_pred,
           'intercept': model.intercept_,
           'coef': model.coef_}
    
    if x_test.shape[1] == 2 and draw:
        # create scatter plot for samples from each class
        for class_value in range(2):
            # get row indexes for samples with this class
            row_ix = np.where(y_pred == class_value)
            # create scatter of these samples
            pyplot.scatter(x_poly_test[row_ix, 0], x_poly_test[row_ix, 1])
        # show the plot
        pyplot.show()
    
    return res
  
def own_svm(x_train, y_train, x_test, 
        C=1, penalty=None, graph=False):
    
    assert C >0, "Сила регуляризации должна быть больше 0!"
    assert len(np.unique(y_train)) > 1, 'В y_train должно быть больше одного класса!'
    
    if x_train.shape[0] < 2**x_train.shape[1]:
        print('Для оптимального результата количество наблюдений должно быть больше 2^k.')
        IsContinue = int(input('Все равно продолжить? (0/1): '))
        if not IsContinue:
            return
        
    if penalty not in ['l1', 'l2']:
        penalty = 'l2'
    
    model = LinearSVC(max_iter=50_000, penalty=penalty, C=C, dual=False)
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    
    res = {'x_test': x_test,
           'y_pred': y_pred,
           'intercept': model.intercept_,
           'coef': model.coef_}
    
    if x_test.shape[1] == 2 and graph:
        for class_value in range(2):
            row_ix = np.where(y_pred == class_value)
            pyplot.scatter(x_test[row_ix, 0], x_test[row_ix, 1])
        pyplot.show()
    
    return res
