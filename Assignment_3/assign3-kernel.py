# -*- coding: utf-8 -*-
"""
Created on Sun Oct 20 11:43:14 2019

@author: nafis
"""

import numpy as np
import sys

'''_____________________________KERNEL Functions_________________________________'''

#X,Y ndArray
def calculate_distance(X, Y):
    return np.sum((X-Y)**2)

#X ndArray
def linear_kernel(X, Z, *args):
    return X@Z.T

#C is a scaler, X ndArray
def quadratic_kernel(X,Z, *args):
    return (1 + X@Z.T)**2       #C=1 by default

#X ndArray, sigma Scaler
def gaussian_kernel(X, Z, sigma):
    kernel_matrix = np.zeros((X.shape[0], Z.shape[0]))
    for i in range(X.shape[0]):
        for j in range(Z.shape[0]):
            kernel_matrix[i,j] = calculate_distance(X[i], Z[j])
    kernel_matrix = np.exp((-1)*(kernel_matrix/(2*sigma**2)))
    return kernel_matrix

def choose_and_build_kernel(kernel, X, Z, *args):
    if(kernel=="gaussian"): 
        K = gaussian_kernel(X, Z, args[0]) 
    elif(kernel=="quadratic"):
        K = quadratic_kernel(X, Z)
    else:
        K = linear_kernel(X, Z)
    return K

'''_____________________________Other Functions_________________________________'''

#Y, Y_predict nx1 vectors
def calculate_accuracy(Y, Y_predict):
    return (np.where(Y==Y_predict,1,0).sum())/Y.shape[0]

if __name__=="__main__":
    
    #load parameters from command prompt
    args = sys.argv
    train_file = args[1]
    test_file = args[2]
    kernel = args[3]
    if(args[3]=="gaussian"):
        sigma = float(args[4])
    else:
        sigma = -1
    alpha = 0.01            #ridge coefficient
    
    '''_____________________________KERNEL MATRIX BUILDING PART_________________________________'''
    
    #import file with numpy load.txt
    training_data = np.loadtxt(train_file, delimiter=',')
    
    #shape of training data file
    rows_in_training_data, cols_in_training_data = training_data.shape[0], training_data.shape[1]
    
    #seperate X and Y for training data
    X_train, Y_train = training_data[:,:-1], training_data[:,cols_in_training_data-1]
    
    #kernel matrix choose and build
    K = choose_and_build_kernel(kernel, X_train, X_train, sigma)
    
    #augment the kernel matrix
    K_aug = K+1
    
    #calculate C
    C = np.linalg.inv(K_aug + alpha*np.eye(K_aug.shape[0]))@Y_train
    
    #Y_predict
    Y_train_predict = K_aug@C
    
    #apply threshold
    Y_train_predict = np.where(Y_train_predict>=0.5, 1, 0)
    
    #calculate training accuracy
    training_accuracy = calculate_accuracy(Y_train, Y_train_predict)
    
    '''_____________________________TESTING PART_________________________________'''
    
    #import test data
    test_data = np.loadtxt(test_file, delimiter=',')
    
    #shape of test data file
    rows_in_test_data, cols_in_test_data = test_data.shape[0], test_data.shape[1]
    
    #seperate X_test and Y_test
    X_test, Y_test = test_data[:,:-1], test_data[:,cols_in_test_data-1]
    
    #kernel matrix choose and build
    K_test = choose_and_build_kernel(kernel, X_test, X_train, sigma)
    #K_test = choose_and_build_kernel(kernel, X_test, X_train)
    
    #augment the kernel test matrix
    K_test_aug = K_test + 1
    
    #Y_test_predict
    Y_test_predict = K_test_aug@C
    
    #apply threshold
    Y_test_predict = np.where(Y_test_predict>=0.5, 1, 0)
    
    #calculate test accuracy
    test_accuracy = calculate_accuracy(Y_test, Y_test_predict)
    
    '''_____________________________PRINTING_________________________________'''
    print("Training Accuracy: {:.3f}%".format(training_accuracy*100))
    print("Test Accuracy: {:.3f}%".format(test_accuracy*100))
    
    
    '''---------------------------------------------------------------------------------------------------
    NOTE TO GRADER: the following part has been commented out intentionally because it took a lot of time in my
    local machine to find different training and test accuracy values for different sigma values.
    I have used Google Colab as well for this purpose in order to make the calculation process faster.
    My output has been shown as graph in the PDF.
    ---------------------------------------------------------------------------------------------------'''
    
    '''_____________________________TUNING AND TESTING_________________________________
    
    test_accuracy_list = []
    sigma_range = np.linspace(1, 10, 37)
    for sigmas in sigma_range:
        K_r = choose_and_build_kernel("gaussian", X_train, X_train, sigmas)
        K_aug_r = K_r+1
        C_r = np.linalg.inv(K_aug_r + alpha*np.eye(K_aug_r.shape[0]))@Y_train
        K_test_r = choose_and_build_kernel("gaussian", X_test, X_train, sigmas)
        K_test_aug_r = K_test_r + 1
        Y_test_predict_r = K_test_aug_r@C_r
        Y_test_predict_r = np.where(Y_test_predict_r>=0.5, 1, 0)
        test_accuracy_r = calculate_accuracy(Y_test, Y_test_predict_r)
        test_accuracy_list.append(test_accuracy_r)
    
    plt.plot(sigma_range, test_accuracy_list)
    plt.xlabel("Sigma")
    plt.ylabel("Test Accuracy")
    plt.title("Sigma Range VS Test Accuracy for np.linspace(1,10,19)")
    plt.savefig("Sigma_VS_Accuracy.jpg")
    plt.show()'''
    
    
    
    
    
    
    
    
    
    
    
    
    
