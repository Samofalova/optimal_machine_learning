import numpy as np
import matplotlib.pyplot as plt

def own_pegasos(X_train, y_train, X_test, lam = 0.001, max_iter = 1000, stoch_size = 0.2, draw = False):
    """
    A function that implements a two-class classification model using
    the pegasos method for the reference vector
    method training problem.
    Parameters
    ----------
    X_train : numpy.array
        A set of features for the training sample.
    y_train : numpy.array
        Set of class values for the training sample (classes can be only -1 or 1).
    X_test : numpy.array
        A set of features for the test sample.
    lam: float, default= 0.001
        the power of regularization
    max_iter : int, default=1000
        Number of learning iterations
    stoch_size: float, default = 0.2
        the proportion of the training sample for stochastic gradient descent
    draw : bool, default=False
        Plotting the classification graph.

    Examples
    >>> from sklearn.datasets import make_blobs
    >>> X, Y = make_blobs(n_samples = 40,centers=2, cluster_std=1.2,n_features=2,random_state=42)
    >>> for i,j in enumerate(Y):
    >>>     if j == 0:
    >>>         Y[i] = -1
    >>>     elif j == 1:
    >>>         Y[i] = 1
    >>> X_train = X[:35]
    >>> Y_train = Y[:35]
    >>> X_test= X[35:]
    >>> Y_test = Y[35:]
    >>> own_pegasos(X_train, Y_train, X_test)
    >>> {'weights': array([ 0.38290961, -0.09323558,  0.05267501]),
    >>> 'y_pred': array([ 1, -1,  1, -1,  1])}
    """
    if len(y_train.shape) != 1 or (np.unique(y_train) == np.array([-1, 1])).sum() != 2:
        raise ValueError('массив классов должен быть одномерным и содержать только два класса: -1 или 1')
    if X_train.shape[1] != X_test.shape[1]:
        raise ValueError('кол-во признаков в обучающей и тестовой выборке должно быть одинаковым')
    k= stoch_size * X_train.shape[0]
    inds =np.arange(X_train.shape[0])
    #добавление столбцов из единиц
    X_train = np.concatenate((X_train, np.ones((X_train.shape[0],1))), axis=1)
    X_test = np.concatenate((X_test, np.ones((X_test.shape[0],1))), axis=1)
    w = np.zeros(X_train.shape[1])
    margin_current = 0
    margin_previous = -10
    not_converged = True
    t = 0

    while(not_converged):
        margin_previous = margin_current
        t += 1
        pos_support_vectors = 0
        neg_support_vectors = 0
        eta = 1/(lam*t)
        fac = (1-(eta*lam))*w
        np.random.shuffle(inds)
        selected_inds = inds[:round(k)]
        for i in selected_inds:
            prediction = np.dot(X_train[i], w)

            if (round((prediction),1) == 1):
                pos_support_vectors += 1
            if (round((prediction),1) == -1):
                neg_support_vectors += 1

            if y_train[i]*prediction < 1:
                w = fac + eta*y_train[i]*X_train[i]
            else:
                w = fac

        if t>max_iter:    
            margin_current = np.linalg.norm(w)
            if((pos_support_vectors > 0)and(neg_support_vectors > 0)and((margin_current - margin_previous) < 0.01)):
                not_converged = False

    y_pred = []
    for i in X_test:
        pred = np.dot(w,i)
        if (pred > 0):
            y_pred.append(1)
        elif (pred < 0):
            y_pred.append(-1)

    if X_test.shape[1] == 3 and draw:
        plt.scatter(X_test[:, 0], X_test[:, 1], c= y_pred, s=20, cmap='viridis')
        plt.xlabel('Признак 1')
        plt.ylabel('Признак 2')
    elif draw:
        raise ValueError('не можем построить график')
    return {'y_pred': np.array(y_pred), 'weights': w}
