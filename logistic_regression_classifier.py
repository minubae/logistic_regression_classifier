###########################################################################################################################
# Title: Logistic Regression Classifier
# Date: 03/08/2019, Friday
# Author: Minwoo Bae (minwoo.bae@uconn.edu)
# Institute: The Department of Computer Science and Engineering, UCONN
###########################################################################################################################
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import inv, norm
from numpy import transpose as T
from numpy import array as vec
from numpy import dot, exp, add, sum, random, abs, log, abs, log, ones, sort, argsort
from sklearn.model_selection import train_test_split

# Compute min logistic likelihood:
def get_log_likelihood(coeff_vec, features_mat, class_vec):

    w = coeff_vec
    X = features_mat
    y = class_vec

    n = len(X)
    log_like_temp = []
    log_likelihood = []

    for i in range(n):

        log_like = log(1+exp((X[i].T).dot(w))) - y[i]*(X[i].T).dot(w)
        log_like_temp.append(log_like)

    log_likelihood = vec(sum(log_like_temp, axis=0))

    return log_likelihood

# Compute a gradient descent algorithm:
def get_gradient(coeff_vec, features_mat, class_vec):

    w_0 = coeff_vec
    X = features_mat
    y = class_vec

    n = len(X)
    gradient_mat = []
    gradient = []

    for i in range(n):

        # logit = 1 / (1+exp(-(w_0.T).dot(X[i])))
        logit = exp(X[i].T.dot(w_0)) / (1+exp(X[i].T.dot(w_0)))
        grad = X[i]*logit - y[i]*X[i]
        gradient_mat.append(grad)

    # print(vec(gradient_mat))
    gradient = vec(sum(gradient_mat, axis=0))

    return gradient

# Compute argmin coefficients w by using the gradient descent:
def get_argmin_w(coeff_vec, features_mat, class_vec):

    w_0 = coeff_vec
    X = features_mat
    y = class_vec

    error = 1e-1 #0.1
    epsilon = 1e-5 #0.00000001
    # step = 10**(-5)
    step = 10**(-6)
    w_temp = w_0
    k = 0

    while epsilon < error:

        w_k = w_temp

        # get gradient of Log Likehood function:
        gradient = get_gradient(w_k, X, y)

        # get the direction of the negative gradient:
        d_k = -gradient
        w_temp = w_k + step*d_k

        log_likelihood_01 = get_log_likelihood(w_temp, X, y)
        log_likelihood_02 = get_log_likelihood(w_k, X, y)
        error = norm(log_likelihood_01 - log_likelihood_02)

        k += 1
        print('error[',k,']:', sum(error))
        print('')

    return w_temp

# Compute a probability by Logistic regression:
def get_logistic_prob(opt_w, features_mat):

    p_y = []
    X_test = features_mat
    W = opt_w

    n = len(X_test)
    for i in range(n):
        # p_i = exp((X_test[i].T).dot(argmin_w)) / (1 + exp((X_test[i].T).dot(argmin_w)))
        p_i = exp((W.T).dot(X_test[i])) / (1 + exp((W.T).dot(X_test[i])))
        p_y.append(p_i)
        # print(p_i)

    p_y = vec(p_y)
    return p_y

def get_sort(data01, data02):
    
    d1 = data01
    d2 = data02
    posterior = []
    y_test = []

    for i in argsort(d1):
        posterior.append(d1[i])
        y_test.append(d2[i])

    return posterior, y_test

def get_ROC_curve(features_mat, class_vec):
    # '''
    X = features_mat
    y = responses
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
    # '''
    '''
    n = 500
    p = 5
    X = random.randint(4, size=[n, p])
    y = random.randint(2, size=n)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
    '''
    p = len(X_train.T)

    # Initial guess for w_0:
    w_0 = random.uniform(0, 0.0001, size=p)

    norm_train = norm(X_train)**(-1)
    norm_test = norm(X_test)**(-1)

    X_train = X_train*norm_train
    X_test = X_test*norm_test

    # Compute the argmin w by using the gradient decesent:
    argmin_w = get_argmin_w(w_0, X_train, y_train)
    posterior = get_logistic_prob(argmin_w, X_test)

    '''
    # Posterior probability: p(decay|data)
                    # 0     0     1     0     1    0     0      1     1     0    1     1     1     1     0     1     1     1
    posterior_01 = [0.03, 0.08, 0.10, 0.11, 0.22, 0.32, 0.35, 0.42, 0.44, 0.48, 0.56, 0.65, 0.71, 0.72, 0.73, 0.80, 0.82, 0.99]
                    # 0     1     0     0     1     1     1     0     1     0    1     1     0     1     1     0     1     1
    posterior_02 = [0.11, 0.56, 0.03, 0.08, 0.82, 0.22, 0.71, 0.32, 0.99, 0.35, 0.42, 0.44, 0.48, 0.65, 0.72, 0.73, 0.80, 0.10]
    # 0 = a (active), 1 = d (decoy)
    y_test_01 = [0,0,1,0,1,0,0,1,1,0,1,1,1,1,0,1,1,1]
    y_test_02 = [0,1,0,0,1,1,1,0,1,0,1,1,0,1,1,0,1,1]

    posterior, y_test = get_sort(posterior_02, y_test_02)
    '''

    posterior, y_test = get_sort(posterior, y_test)

    print('posterior:')
    print(posterior)
    print('y_test:')
    print(y_test)

    n = len(posterior)

    data_roc = []
    TPR_vec = []
    FPR_vec = []
    Precision_vec = []
    Sensitivity_vec = []
    Specificity_vec = []

    for i, t in enumerate(posterior):
        # print(x)
        TP = 0; FP = 0
        TN = 0; FN = 0
        TPR = 0; FPR = 0
        Precision = 0; Sensitivity = 0
        Specificity = 0

        for j in range(i, n):
            # print(actual_class[j])
            if y_test[j] == 1:
                TP+=1
            else:
                FP+=1
        for k in reversed(range(i)):
            # print('k:',k)
            # print(actual_class[k])
            if y_test[k] == 0:
                TN+=1
            else:
                FN+=1

        TPR = TP/(TP+FN)
        TPR_vec.append(TPR)

        FPR = FP/(FP+TN)
        FPR_vec.append(FPR)

        Precision = TP/(TP+FP)
        Precision_vec.append(Precision)

        Sensitivity = TP/(TP+FN)
        Sensitivity_vec.append(Sensitivity)

        Sepcificity = TN/(TN+FP)
        # Specificity_vec.append(Sepcificity)
        Specificity_vec.append(1-Sepcificity)
        '''
        print('TP: ', TP)
        print('FP: ', FP)
        print('TN: ', TN)
        print('FN: ', FN)
        print('')
        '''

    return TPR_vec, FPR_vec, Precision_vec, Sensitivity_vec, Specificity_vec


def main(features_mat, class_vec):

    TPRc, FPRc, precision, sensitivity, specificity = get_ROC_curve(data, responses)
    # print(TPRc)
    # print(FPRc)
    # '''
    plt.title('ROC curve')
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    # plt.plot(FPRc, TPRc, 'r', marker='.')
    plt.plot(TPRc, FPRc, 'r', marker='.')
    # plt.plot(FPRc, TPRc, 'r', marker='o')
    # plt.plot(FPRc, TPRc, 'r')
    # plt.plot(specificity, sensitivity, 'r')
    plt.axis([0, 1, 0, 1])
    plt.show()

    # print(vec(Precision_vec).T)
    # print(vec(Sensitivity_vec).T)
    # print(vec(Specificity_vec).T)

if __name__ == '__main__':

    from sklearn.datasets import load_breast_cancer

    #Given data matrices:
    data_mat = load_breast_cancer()
    # data = data_mat.data[:50]
    # responses = data_mat.target[:50]
    data = data_mat.data
    responses = data_mat.target

    main(data, responses)
