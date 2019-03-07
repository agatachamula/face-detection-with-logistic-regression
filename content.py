
import numpy as np

def sigmoid(x):
    '''
    :param x: input vector Nx1
    :return: vector of sigmoid function values calculated for elements x, Nx1
    '''
    print(x.shape)
    N=x.shape[0]
    result=np.zeros((N,1))
    result=1/(1+np.exp(-x))

    print(result.shape)


    return result


def logistic_cost_function(w, x_train, y_train):
    '''
    :param w: model parameters Mx1
    :param x_train: training set features NxM
    :param y_train: training set labels Nx1
    :return: function returns tuple (val, grad), where val is a velue of logistic function and grad its gradient over w
    '''
    N=x_train.shape[0]
    M=x_train.shape[1]

    multi=1
    sig = sigmoid(x_train @ w)

    for n in range (0,N):
        multi=multi*(sig[n]**(y_train[n]) * (1-sig[n])**(1-y_train[n]))


    multi=(-1/N)*np.log(multi)

    grad = -((np.transpose(x_train) @ (y_train-sig))/N)

    return(multi[0],grad)


def gradient_descent(obj_fun, w0, epochs, eta):
    '''
    :param obj_fun: objective function that is going to be minimized (call val,grad = obj_fun(w)).
    :param w0: starting point Mx1
    :param epochs: number of epochs / iterations of an algorithm
    :param eta: learning rate
    :return: function optimizes obj_fun using gradient descent. It returns (w,func_values),
    where w is optimal value of w and func_valus is vector of values of objective function [epochs x 1] calculated for each epoch
    '''

    func_values=np.zeros((epochs,1))
    start=w0
    grad = obj_fun(start)[1]

    for i in range (0,epochs):
        best= start - eta*grad
        func_values[i], grad =obj_fun(best)
        start=best


    return(best,func_values)



def stochastic_gradient_descent(obj_fun, x_train, y_train, w0, epochs, eta, mini_batch):
    '''
    :param obj_fun: objective function that undergoes optimization. Call val,grad = obj_fun(w,x,y), where x,y indicates mini-batches.
    :param x_train: training data (feature vectors)NxM
    :param y_train: training data (labels) Nx1
    :param w0: starting point Mx1
    :param epochs: number of epochs
    :param eta: learning rate
    :param mini_batch: size of mini-batches
    :return: function optimizes obj_fun using gradient descent. It returns (w,func_values),
    where w is optimal value of w and func_valus is vector of values of objective function [epochs x 1] calculated for each epoch. V
    Values of func_values are calculated for entire training set!
    '''
    start=w0
    func_values=np.zeros((epochs, 1))
    N=x_train.shape[0]//mini_batch
    start=w0

    for i in range(0,epochs):
        data_start=0
        for j in range(0,N):
            Xdata=x_train[range(data_start, data_start+mini_batch), :]
            Ydata=y_train[range(data_start, data_start+mini_batch), :]
            best= start - eta*obj_fun(start,Xdata,Ydata)[1]
            start=best
            data_start=data_start+mini_batch
        func_values[i] = obj_fun(best, x_train, y_train)[0]

    return (best, func_values)


def regularized_logistic_cost_function(w, x_train, y_train, regularization_lambda):
    '''
    :param w: model parameters Mx1
    :param x_train: training set (features) NxM
    :param y_train: training set (labels) Nx1
    :param regularization_lambda: regularization parameters
    :return: function returns tuple(val, grad), where val is a velue of logistic function with regularization l2,
    and grad its gradient over w
    '''

    M=w.shape[0]
    val=logistic_cost_function(w,x_train,y_train)[0]
    grad = logistic_cost_function(w, x_train, y_train)[1]

    for i in range(1,M):
        val=val + (regularization_lambda/2) * (w[i]**2)
        grad[i,0]= grad[i,0] + regularization_lambda * (w[i])


    return(val[0],grad)


def prediction(x, w, theta):
    '''
    :param x: observation matrix NxM
    :param w: parameter vector Mx1
    :param theta: classification threshold [0,1]
    :return: function calculates vector y Nx1. Vector is composed of labels {0,1} for observations x
     calculated using model (parameters w) and classification threshold theta
    '''

    N=x.shape[0]
    y=np.zeros((N,1))
    sigm=sigmoid(w.transpose()@x.transpose())

    for i in range(0,N):
        if(sigm[0,i]>=theta):
            y[i]=1

    return y



def f_measure(y_true, y_pred):
    '''
    :param y_true: vector of ground truth labels Nx1
    :param y_pred: vector of predicted labels Nx1
    :return: value of F-measure
    '''
    N=y_true.shape[0]

    TP=0
    FP=0
    FN=0

    for i in range(0,N):
        if(y_true[i]==1 and y_pred[i]==1):
            TP=TP+1
        elif(y_true[i]==0 and y_pred[i]==1):
            FP=FP+1
        elif(y_true[i]==1 and y_pred[i]==0):
            FN=FN+1

    F=(2*TP)/(2*TP + FP + FN)

    return F


def model_selection(x_train, y_train, x_val, y_val, w0, epochs, eta, mini_batch, lambdas, thetas):
    '''
    :param x_train: trainig set (features) NxM
    :param y_train: training set (labels) Nx1
    :param x_val: validation set (features) Nval x M
    :param y_val: validation set (labels) Nval x 1
    :param w0: vector of initial values of w
    :param epochs: number of iterations of SGD
    :param eta: learning rate
    :param mini_batch: mini-batch size
    :param lambdas: list of lambda values that have to be considered in model selection procedure
    :param thetas: list of theta values that have to be considered in model selection procedure
    :return: Functions makes a model selection. It returs tuple (regularization_lambda, theta, w, F), where regularization_lambda
    is the best velue of regularization parameter, theta is the best classification threshold, and w is the best model parameter vector.
    Additionally function returns matrix F, which stores F-measures calculated for each pair (lambda, theta).
    Use SGD and training criterium with l2 regularization for training.
    '''

    L=len(lambdas)
    T=len(thetas)
    M=x_train.shape[1]

    Fs=np.zeros((L,T))
    w=[]

    for i in range(0,L):
        obj_fun = lambda w, x, y: regularized_logistic_cost_function(w, x, y, lambdas[i])
        w.append(stochastic_gradient_descent(obj_fun, x_train, y_train, w0,epochs,eta,mini_batch)[0])
        for j in range(0, T):
            y_pred=prediction(x_val,w[i],thetas[j])
            Fs[i,j]=f_measure(y_val,y_pred)

    flatArrayPos = np.argmax(Fs)
    indexT = flatArrayPos % T
    indexL = flatArrayPos // T

    bestL = lambdas[indexL]
    bestT = thetas[indexT]



    return(bestL,bestT,w[indexL],Fs)
