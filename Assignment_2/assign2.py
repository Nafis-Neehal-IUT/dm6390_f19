# -*- coding: utf-8 -*-
"""
Created on Wed Oct  9 00:14:20 2019

@author: nafis
"""

import numpy as np
import sys
import matplotlib.pyplot as plt

def Project(X, U):                                  #projection of X on U
    return (X.T@U)/(U.T@U)

def QR_factorize(D):
    Q = np.zeros_like(D)                            #same size as D (will have U's as column vectors from 0...d)
    R = np.eye(D.shape[1])
    #QR Factorization here
    for i in range(D.shape[1]):                     #for each columns
        if i==0:
            Q[:,i] = D[:,i]                         #for first column
            
        else:
            sum_col = np.zeros_like(Q[:,i])
            for j in range(0, i):                   #j will be 0....d-1 at max
                #projections of data column (attribute) on each U column
                proj_val = Project(D[:,i],Q[:,j])
                sum_col = sum_col + proj_val * Q[:,j]
                R[j,i] = proj_val
            Q[:,i] = D[:,i] - sum_col
            
    return Q, R
    
def Calc_del_inv(D, Q):
    del_inv = np.zeros((D.shape[1], D.shape[1]))
    for i in range(Q.shape[1]):
        del_inv[i,i] = 1/(Q[:,i].T@Q[:,i])        
    return del_inv

def Backsolve_w(D, Q, R, Y, del_inv):
    W = np.zeros((D.shape[1]))
    B = del_inv@Q.T@Y
    for i in range((D.shape[1]-1), -1, -1):
        if i==(D.shape[1]-1):
            W[i] = B[i]
        else:
            sum_pw = 0
            for j in range((D.shape[1]-1),i, -1): 
                sum_pw = sum_pw + R[i, j]*W[j]
            W[i] = B[i] - sum_pw
    return W

def Calc_SSE(Y1, Y2):
    return np.sum((Y1-Y2)**2)

def Calc_TSS(Y):
    return np.sum((Y-np.mean(Y))**2)

def Calc_R_Sq(SSE, TSS):
    return (TSS-SSE)/TSS

def Implement_train(X, Y):
    
    #part 1 implementation
    D = np.hstack((np.ones((X.shape[0],1)), X))     #augment the dataset with ones (nxd+1)
    Q, R = QR_factorize(D)
    
    #calculating del_inv
    del_inv = Calc_del_inv(D, Q)
    
    #backsolve for W 
    W = Backsolve_w(D, Q, R, Y, del_inv)
    
    #calculating Y'
    Y_predict = D@W
    
    #SSE for training data
    SSE = Calc_SSE(Y, Y_predict) 
    
    #TSS for training data
    TSS = Calc_TSS(Y)
    
    #R^2 calculation
    R_Sq = Calc_R_Sq(SSE, TSS)
    
    return D, W, SSE, R_Sq

def Implement_Ridge(D, Y):
    #part 2 implementation
    
    Q, R = QR_factorize(D)
    
    #calculating del_inv
    del_inv = Calc_del_inv(D, Q)
    
    #backsolve for W 
    W = Backsolve_w(D, Q, R, Y, del_inv)
    
    #calculating Y'
    Y_predict = D@W
    
    #SSE for training data
    SSE = np.sum((Y-Y_predict)**2)
    
    #TSS for training data
    TSS = np.sum((Y-np.mean(Y))**2)
    
    #R^2 calculation
    R_Sq = (TSS-SSE)/TSS
    
    return D, W, SSE, R_Sq    

if __name__=='__main__':
    
    '''IMPORT Files'''
    args = sys.argv
    train = args[1]
    test = args[2]
    ridge = np.float32(args[3])
    
    '''PART 1'''
    
    '''APPLY ON TRAIN DATA'''
    
    #import train data
    data_train = np.loadtxt(train, delimiter=',')
    X_train = data_train[:,:-1]
    Y_train = data_train[:,data_train.shape[1]-1]
    
    #apply on train data
    D_train, W_train, SSE_train, R_Sq_train = Implement_train(X_train, Y_train)
    
    '''APPLY ON TEST DATA'''
    #import test data
    data_test = np.loadtxt(test, delimiter=',')
    X_test = data_test[:,:-1]
    Y_test = data_test[:,data_test.shape[1]-1]
    
    #apply on test data
    D_test = np.hstack((np.ones((X_test.shape[0],1)), X_test))
    Y_test_predict = D_test@W_train
    
    #calculate performance
    #SSE for test data
    SSE_test = Calc_SSE(Y_test, Y_test_predict)
    
    #TSS for training data
    TSS_test = Calc_TSS(Y_test)
    
    #R^2 calculation
    R_Sq_test = Calc_R_Sq(SSE_test, TSS_test)
    
    
    '''PART 2 - WITH RIDGE CONSTANT'''
    
    #parameters
    alpha = ridge
    I = np.eye(data_train.shape[1])
    A = alpha*I
    
    #new data matrix build
    D_ridge_train = np.vstack((D_train, np.sqrt(A)))
    Y_ridge_train = np.concatenate((Y_train, np.zeros_like(A[0])), axis=0)
    
    D_ridge_test = np.vstack((D_test, np.sqrt(A)))
    Y_ridge_test = np.concatenate((Y_test, np.zeros_like(A[0])), axis=0)
    
    
    D_ridge_train, W_ridge_train, SSE_ridge_train, R_Sq_Ridge_train = Implement_Ridge(D_ridge_train, Y_ridge_train)
    
    Y_ridge_test_predict = D_ridge_test@W_ridge_train
    
    #calculate performance
    #SSE for test data
    SSE_ridge_test = Calc_SSE(Y_ridge_test, Y_ridge_test_predict)
    
    #TSS for training data
    TSS_ridge_test = Calc_TSS(Y_ridge_test)
    
    #R^2 calculation
    R_Sq_ridge_test = Calc_R_Sq(SSE_ridge_test, TSS_ridge_test)
    
    #print stuff (for ridge only)
    #ridge on training
    print("Ridge Regression on Training Data-")
    print("Weight Vector: ", np.around(W_ridge_train, decimals=3))
    print("L2 Norm of W Vector: {:.3f}".format(np.sqrt(W_ridge_train.T@W_ridge_train)))
    print("SSE Value for Training: {:.3f}".format(SSE_ridge_train))
    print("R-Squared Value for Training: {:.3f}".format(R_Sq_train))
    print("")
    print("Ridge Regression Performance on Test Data-")
    print("SSE Value for Ridge Test: {:.3f}".format(SSE_ridge_test))
    print("R-Squared Value for Ridge Test: {:.3f}".format(R_Sq_ridge_test))
    
    
    '''PART 3'''
    #RIDGE Value VS W Test
    
    '''Test Values
    
    alpha_t = 10
    W = [-11.481   0.556   0.088   0.084   0.018   0.069  -1.454   0.015   1.345
  -0.627   0.965  -0.073]
    
    alpha_t = 5
    W = [-13.785   0.556   0.088   0.084   0.018   0.069  -1.453   0.017   1.345
  -0.627   0.967  -0.073]
    
    
    alpha_t = 1e0
    W = [-16.421   0.556   0.088   0.084   0.018   0.069  -1.451   0.02    1.346
     -0.626   0.969  -0.073]
    
    alpha_t = 1e-1
    W = [-17.159   0.556   0.088   0.085   0.018   0.069  -1.451   0.021   1.346
  -0.626   0.97   -0.073]
    
    alpha_t = 1e-2
    W = [-17.237   0.556   0.088   0.085   0.018   0.069  -1.451   0.021   1.346
  -0.626   0.97   -0.073]
    
    alpha_t = 1e-3
    W = [-17.245   0.556   0.088   0.085   0.018   0.069  -1.451   0.021   1.346
  -0.626   0.97   -0.073]
    '''

    
        
            
            