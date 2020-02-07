# -*- coding: utf-8 -*-
"""
Created on Sun Nov 17 14:02:53 2019

@author: nafis
"""

#imports
import numpy as np
import math
import sys

#function alias
p = print

#kernel functions and selector
'''
The following 3 functions are kernel functions. Gaussian takes Sigma as extra
argument.
'''
def linear_kernel(X, Z, *args):
    return X@Z.T
    
def h_quadratic_kernel(X, Z, *args):
    return (X@Z.T)**2

def gaussian_kernel(X,Z, sigma):
    kernel_matrix = np.array([np.sum((X[i]-Z[j])**2) for i in range(X.shape[0])
                     for j in range(Z.shape[0])])
    kernel_matrix = np.exp(((-1)*kernel_matrix)/(2*(sigma**2)))
    kernel_matrix = kernel_matrix.reshape(X.shape[0], Z.shape[0])
    return kernel_matrix

'''
The following function will select which Kernel to use based on user input and
call that function accordingly.
'''
def choose_and_build_kernel(kernel, X, Z, *args):
    if(kernel=='gaussian'):
        K = gaussian_kernel(X, Z, args[0])
    elif(kernel=='quadratic'):
        K = h_quadratic_kernel(X, Z)
    else:
        K = linear_kernel(X, Z)
    return K

'''
The following function will calculate the step size matrix
'''
def calculate_step_size(K):
    E = np.array([1/K[i,i] for i in range(K.shape[0])])
    return E

'''
The following function will calculate Del Alpha
'''
def calculate_alpha_update(Yk, alpha, Y, K):
    n = Y.shape[0]
    sum_value = 0
    for i in range(n):
        sum_value += alpha[i]*Y[i]*K[i]
    return Yk*sum_value

'''
The following function will calculate squarred error between two vectors
'''
def calculate_error(a, a_old):
    return np.sqrt(np.sum((a-a_old)**2))

'''
The following function will calculate Alpha
'''
def calculate_alpha(trainY, Kernel, Eta, C, eps):
    Alpha = np.zeros_like(trainY)
    epoch = 0
    while(True):
        
        Alpha_old = np.copy(Alpha)
        
        for k in range(1, trainY.shape[0]):
            Alpha[k] += Eta[k]*(1-calculate_alpha_update(trainY[k], Alpha, trainY, 
                                                     Kernel[:,k]))
            if Alpha[k]<0:
                Alpha[k]=0
            if Alpha[k]>C:
                Alpha[k]=C
        
        error = calculate_error(Alpha, Alpha_old)
        if error<=eps or epoch>=1000:   #both time and error bound
            break
        epoch = epoch+1
    return Alpha

'''
The following function will calculate Combination nCr
'''
def nCr(n,r):
    f = math.factorial
    return f(n) / f(r) / f(n-r)

'''
The following function f(Z) will return +1 if Z>=0 or -1 if Z<0
'''
def Sign(X):
    X[X>=0]=1
    X[X<0]=-1
    return X

'''
The following function will calculate the predicted response value
'''
def calculate_prediction(Alpha, Y, KernelTest):
    prediction = np.zeros_like(Y)
    
    idxs = np.flatnonzero(Alpha>0)
    
    for j in range(Y.shape[0]):
        for idx in idxs:
            prediction[j]+= Alpha[idx] * trainY[idx] * KernelTest[idx, j]
            
    return prediction

#--------------------------------------------------------------------------------
if __name__ == '__main__':
    
    #take arguments from command prompt
    args = sys.argv
    trainFile = args[1]
    testFile = args[2]
    C = float(args[3])
    eps = float(args[4])
    kernel = args[5]
    sigma = float(args[6])
    COMMA = ','

    #----------------------------------------------------------------------------
    '''
    The following segment will import train and test data 
    (Normalized 0-1, 8 features, 1 binary class -1/1)
    Then seperate it into trainX, trainY, testX, testY
    '''
    
    train = np.genfromtxt(trainFile, delimiter = COMMA, dtype=np.float64)
    test = np.genfromtxt(testFile, delimiter = COMMA, dtype=np.float64)
    
    trainX = train[...,:-1]
    trainY = train[...,-1]
    
    testX = test[...,:-1]
    testY = test[...,-1]
    
    #----------------------------------------------------------------------------
    '''
    The following segment will train on training data to generate final alpha
    and then calculate training accuracy on training data
    '''
    
    #Choose and Calculate kernel matrix and augment (linear/h-quadratic/gaussian)
    Kernel = choose_and_build_kernel(kernel, trainX, trainX, sigma) + 1
    
    #build step size matrix
    Eta = calculate_step_size(Kernel)
    
    #Final alpha
    Alpha = calculate_alpha(trainY, Kernel, Eta, C, eps)
    
    #train predict and accuracy calculate
    trainY_predict = calculate_prediction(Alpha, trainY, Kernel)
    trainY_predict = Sign(trainY_predict)
    train_accuracy = np.sum([trainY==trainY_predict])/(trainY.shape[0])
    
    #build kernel for test data
    KernelTest = choose_and_build_kernel(kernel, trainX, testX, sigma)+1
    
    #Test predict and accuracy calculate
    testY_predict = calculate_prediction(Alpha, testY, KernelTest)
    testY_predict = Sign(testY_predict)
    test_accuracy = np.sum([testY==testY_predict])/(testY.shape[0])
    
    #-----------------------------------------------------------------------------
    '''
    The following segment will calculate weight vector using X values mapped in
    feature space
    '''
    idxs = np.flatnonzero(Alpha>0)
    
    if kernel == 'linear':
        Phi_X = np.hstack((trainX,np.ones_like(trainY)[...,np.newaxis]))
        W = np.array([Alpha[idx]*trainY[idx]*Phi_X[idx] for idx in idxs])
        
    elif kernel == 'quadratic':
        q = 2
        d = trainX.shape[1]
        samples = trainX.shape[0]
        W_size = int(nCr(d+1, q))+1
        PhiX_sqr = np.array([trainX[i]**2 for i in range(samples)])
        PhiX_sqrt = np.array([np.sqrt(2)*trainX[k,i]*trainX[k,j] for k in range(samples) 
                           for i in range(d-1) for j in range(i+1,d)])
        PhiX_sqrt = PhiX_sqrt.reshape(samples, W_size-d-1)
        Phi_X = np.hstack((PhiX_sqr, PhiX_sqrt, np.ones_like(trainY)[...,np.newaxis]))
        W = np.array([Alpha[idx]*trainY[idx]*Phi_X[idx] for idx in idxs])
    
    W = np.sum(W, axis=0)
    
    #-----------------------------------------------------------------------------
    '''
    The following segment will do all the printing.
    '''
    np.set_printoptions(precision=3)
    print("Mapped Weights:", W)
    print("Support Vector indices:", idxs)
    print("Alpha Values:", Alpha[idxs])
    print("Training Accuracy:", train_accuracy)
    print("Test Accuracy:", test_accuracy)
             
                
    
    
        
    
    
    
    
    
    
    
    
    
    
    
    
    