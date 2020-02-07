
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 19 00:50:47 2019

@author: nafis
"""

import numpy as np
import sys


def calculate_error(Y, Y_predict):
    return Y - Y_predict

def sigmoid(Z):
    return 1/(1+np.exp(-Z))

def calculate_distance(X,Y):
    return np.sqrt(np.sum((X-Y)**2))

def calculate_accuracy(Y, Y_predict):
    return (np.where(Y==Y_predict,1,0).sum())/Y.shape[0]

def TRAIN_(W, eta, eps):
    epoch = 0
    while(True):
        W_copy = W
        epoch = epoch + 1
        for i in row_indices_random:
            gradient_W = calculate_error(Y_train[i],
                sigmoid(np.dot(W,X_train_aug[i])))*X_train_aug[i]
            W = W + eta * gradient_W
        error = calculate_distance(W, W_copy)
        if error<=eps:
            break
    return W, epoch

if __name__ == '__main__':
    
    #value from command line
    args = sys.argv
    train_file = args[1]
    test_file = args[2]
    eps = float(args[3])
    eta = float(args[4])    
    
    '''_____________________________TRAINING PART_________________________________'''
    
    #import file with numpy load.txt
    training_data = np.loadtxt(train_file, delimiter=',')
    
    #shape of training data file
    rows_in_training_data, cols_in_training_data = training_data.shape[0], training_data.shape[1]
    
    #seperate X and Y for training data
    X_train, Y_train = training_data[:,:-1], training_data[:,cols_in_training_data-1]
    
    #augment training data X_train
    X_train_aug = np.hstack((np.ones_like(Y_train.reshape(rows_in_training_data,1)), X_train))
    
    #initialize W to zero vector (shape = X_train_aug)
    W = np.zeros(cols_in_training_data)
    
    #randomize row indices
    row_indices_random = np.arange(rows_in_training_data)
    np.random.shuffle(row_indices_random)
    
    #gradient -- TEST
    gradient_W = np.zeros(cols_in_training_data)
    
    #start training
    W, epochs = TRAIN_(W, eta, eps)
    
    #create Y_train_predict
    Y_train_predict = sigmoid(X_train_aug@W)
    
    #apply threshold
    Y_train_predict = np.where(Y_train_predict>=0.5, 1, 0)
    
    #calculate training accuracy
    training_accuracy = calculate_accuracy(Y_train, Y_train_predict)
            
    '''_____________________________TEST PART_________________________________'''
            
    #import test data
    test_data = np.loadtxt(test_file, delimiter=',')
    
    #shape of test data file
    rows_in_test_data, cols_in_test_data = test_data.shape[0], test_data.shape[1]
    
    #seperate X_test and Y_test
    X_test, Y_test = test_data[:,:-1], test_data[:,cols_in_test_data-1]
    
    #augment test data X_test
    X_test_aug = np.hstack((np.ones_like(Y_test.reshape(rows_in_test_data,1)), X_test))
    
    #build Y_test_predict
    Y_test_predict = sigmoid(X_test_aug@W)
    
    #apply threshold
    Y_test_predict = np.where(Y_test_predict>=0.5, 1, 0)
    
    #calculate test accuracy
    test_accuracy = calculate_accuracy(Y_test, Y_test_predict)
    
    '''_____________________________PRINT PART_________________________________'''
    
    np.set_printoptions(precision=3)
    print("Weight Vector:", W)
    print("Training Accuracy: {:.3f}%".format(training_accuracy*100))
    print("Test Accuracy: {:.3f}%".format(test_accuracy*100))
    
    
    
    '''---------------------------------------------------------------------------------------------------
    NOTE TO GRADER: the following part has been commented out intentionally because it took a lot of time in my
    local machine to find different training and test accuracy values for different eta and eps values.
    I have used Google Colab as well for this purpose in order to make the calculation process faster.
    My output has been shown as graph in the PDF.
    Maximum Test Accuracy: 85.92% for [eta = 1e-3 and eps = 1e-6]
    ---------------------------------------------------------------------------------------------------'''
    
    '''______________PLOT PART (for different eta and eps values)______________
    
    
    #different hyperparameter settings
    eta_list = np.geomspace(1e-4, 1e-1, num=4)
    eps_list = np.geomspace(1e-4, 1e-1, num=4)
    
    #lists to save different values
    epoch_list = []
    W_list = []
    training_accuracy_list = []
    test_accuracy_list = []
    
    for eta_s in eta_list:
        for eps_s in eps_list:
            
            #hold W for different settings
            W_test = np.zeros_like(W)
            
            #train data with new W
            W_res, epoch_res =TRAIN_(W_test, eta_s, eps_s)
            
            #add new W and number of epochs to train
            W_list.append(W_res)
            epoch_list.append(epoch_res)
            
            #calculate prediction using new W on training and test to evaluate performance
            Y_train_result = sigmoid(X_train_aug@W_res)
            Y_train_result = np.where(Y_train_result>=0.5, 1, 0)
            Y_test_result = sigmoid(X_test_aug@W_res)
            Y_test_result = np.where(Y_test_result>=0.5, 1, 0)
            
            #calculate performance -> training and test accuracy for this W
            training_accuracy_list.append(calculate_accuracy(Y_train, Y_train_result))
            test_accuracy_list.append(calculate_accuracy(Y_test, Y_test_result))
            
    #actual plot
    plt.figure()

    plt.subplot(221)
    plt.plot(eps_list, test_accuracy_list[0:4])
    plt.xlabel("Epsilon")
    plt.ylabel("Test Accuracy")
    plt.title('Eta = 1e-4')
    plt.grid(True)
    
    plt.subplot(222)
    plt.plot(eps_list, test_accuracy_list[4:8])
    plt.xlabel("Epsilon")
    plt.ylabel("Test Accuracy")
    plt.title('Eta = 1e-3')
    plt.grid(True)
    
    plt.subplot(223)
    plt.plot(eps_list, test_accuracy_list[8:12])
    plt.xlabel("Epsilon")
    plt.ylabel("Test Accuracy")
    plt.title('Eta = 1e-2')
    plt.grid(True)
    
    plt.subplot(224)
    plt.plot(eps_list, test_accuracy_list[12:16])
    plt.xlabel("Epsilon")
    plt.ylabel("Test Accuracy")
    plt.title('Eta = 1e-1')
    plt.grid(True)
    
    
    plt.subplots_adjust(top=1.5, bottom=0.08, left=0.10, right=0.95, hspace=1.5,
                        wspace=0.5)
    plt.show()'''       
            
            
            
            
            
            
            
            
            
            