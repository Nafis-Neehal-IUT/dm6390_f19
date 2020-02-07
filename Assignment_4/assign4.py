# -*- coding: utf-8 -*-
"""
Created on Thu Oct 24 00:13:46 2019

@author: nafis
"""

import numpy as np
import sys

#will take 8x1 vector as input and output the same dim vector
def ReLU(Z):
    return np.maximum(Z, 0)

#will take 8x1 vector as input and output the same dim vector
def DReLU(Z):
    Z[Z<=0] = 0
    Z[Z>0] = 1
    return Z

#will take 7x1 vector as input and output the same dim vector
def Softmax(X, theta = 1.0, axis = None):
    # make X at least 2d
    y = np.atleast_2d(X)

    # find axis
    if axis is None:
        axis = next(j[0] for j in enumerate(y.shape) if j[1] > 1)

    # multiply y against the theta parameter,
    y = y * float(theta)

    # subtract the max for numerical stability
    y = y - np.expand_dims(np.max(y, axis = axis), axis)

    # exponentiate y
    y = np.exp(y)

    # take the sum along the specified axis
    ax_sum = np.expand_dims(np.sum(y, axis = axis), axis)

    # finally: divide elementwise
    p = y / ax_sum

    # flatten if X was 1D
    if len(X.shape) == 1: p = p.flatten()

    return p

def NMS(A):
    B = np.zeros_like(A)
    B = (A == A.max(axis=1)[:,None]).astype(int)
    return B

def calculate_accuracy(Y, Y_predict):
    Y_predict = NMS(Y_predict.T)
    counter = 0
    for i in range(Y.shape[0]):
        if np.array_equal(Y[i], Y_predict[i]):
            counter += 1
    return counter/Y.shape[0]
    

if __name__ == "__main__":
    
    #take arguments from command line
    args = sys.argv
    train = args[1]
    test = args[2]
    m = int(args[3])
    eta = float(args[4])
    epoch = int(args[5])
    
    #import training data
    training_data = np.loadtxt(train, delimiter=',')
    X = training_data[:,:-1]
    Y = training_data[:,-1]
    
    #hyperparameters of the model
    t = 1
    num_classes = np.unique(Y).shape[0]         #size of output layer - 7
    num_attributes = X.shape[1]                 #size of input layer - 9
    
    #randomize row indices
    row_indices_random = np.arange(X.shape[0])
    np.random.shuffle(row_indices_random)
    
    #initialize B_o, B_h, W_o, W_h
    B_h = np.random.rand(m,1)                   #8x1
    B_o = np.random.rand(num_classes,1)         #7x1
    W_h = np.random.rand(num_attributes, m)     #9x8
    W_o = np.random.rand(m,num_classes)         #8x7
    
    #one hot encoded Y
    OHE_Map = np.eye(num_classes)               #7x7 identity matrix, one row for each class
    Y_ohe = np.zeros((X.shape[0],num_classes))

    for i in range(Y.shape[0]):
        Y_ohe[i] = OHE_Map[int(Y[i]-1)] 

    #TRAINING
    
    while(t<=epoch):    
        for i in row_indices_random:
            
            #input
            input_X = X[i].T                                            #9x1
           
            #feed forward
            Z = ReLU(B_h + np.dot(W_h.T, input_X).reshape(m,1))         #8x1
            O = Softmax(B_o + np.dot(W_o.T, Z).reshape(num_classes,1), axis=0)  #7x1
            
            #calculate net gradients
            delta_o = O-Y_ohe[i].reshape(num_classes, 1)                #7x1
            delta_h = np.multiply(DReLU(B_h + np.dot(W_h.T, input_X).reshape(m,1)),
                                  (np.dot(W_o,delta_o)))                #8x1
            
            #GD for bias vectors
            Del_B_o = delta_o
            B_o -= eta*Del_B_o
            Del_B_h = delta_h
            B_h -= eta*Del_B_h
            
            #GD for weight matrices
            Del_W_o = np.dot(Z,delta_o.T)
            W_o -= eta*Del_W_o
            Del_W_h = np.dot(input_X.reshape(num_attributes,1), delta_h.T)
            W_h -= eta*Del_W_h
            
        t = t+1        
    
    #import test file
    test_data = np.loadtxt(test, delimiter=',')
    X_test = test_data[:,:-1]
    Y_test = test_data[:,-1]
    
    #one hot encoding of test response
    Y_test_ohe = np.zeros((X_test.shape[0],num_classes))
    for i in range(Y_test.shape[0]):
        Y_test_ohe[i] = OHE_Map[int(Y_test[i]-1)]
        
    #generate final prediction for training data
    Z_train = ReLU(B_h + np.dot(W_h.T, X.T))    #8x4513
    O_train = Softmax(B_o + np.dot(W_o.T, Z_train)) #7x4513
        
    #generate prediction for test data using Forward Prop
    Z_test = ReLU(B_h + np.dot(W_h.T, X_test.T))    #8x14500
    O_test = Softmax(B_o + np.dot(W_o.T, Z_test))   #7x14500
    
    #training accuracy
    training_accuracy = calculate_accuracy(Y_ohe, O_train)
    
    #test accuracy 
    test_accuracy = calculate_accuracy(Y_test_ohe, O_test)
    
    #print
    print("Weight Vector for Input to Hidden: ", W_h)
    print("Weight Vector for Hidden to Output: ", W_o)
    print("Bias Vector for Input to Hidden: ", B_h)
    print("Bias Vector for Hidden to Output: ", B_o)
    print("Training Accuracy {:.2f}%".format(training_accuracy*100))
    print("Test Accuracy {:.2f}%".format(test_accuracy*100))

        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
            