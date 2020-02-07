# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 01:23:43 2019

@author: nafis
"""
#==============================================================================
# IMPORTS
import numpy as np
from sklearn.metrics import confusion_matrix
import sys

#==============================================================================
# HELPER FUNCTIONS
"""
This function will load all the data as string and then convert it to necessary
datatypes
"""
def import_and_load_data(filename):
    data = np.loadtxt(filename, delimiter=',', dtype=np.str_)
    X = np.array(data[:,:-1],dtype=np.float_)
    y = np.array(data[:,-1])
    return X,y

"""
This function will map string labels to unique integer class labels
"""
def label_encoder(data):
    lookupTable, indexedDataset = np.unique(data, return_inverse=True)
    return lookupTable, indexedDataset

"""
*This function will randomly initialize Mu. 
*kxd vector. 
"""
def initialize_mu(X,k):
    low = 0
    high = X.shape[0]
    d = X.shape[1]
    
    Mu = np.zeros((k, d))
    
    for i in range(k):
        index = np.random.randint(low=low, high=high)
        Mu[i] = X[index]
        
    return Mu

"""
This function will initialize Sigma with Identitiy vector + regularization
constraint 1e-3.
"""
def initialize_sigma(X,k):
    d = X.shape[1]
    Sigma = [np.eye(d) + (1e-3)*np.eye(d) for i in range(k)] 
    return Sigma
    
"""
This function will initialize Prior with 1/num_of_classes.
"""
def initialize_prior(k):
    Prior = np.zeros((k,1))
    Prior.fill(1/k)
    return Prior

"""
This function will calculate normal using Mu and Sigma value
"""
def calculate_normal(x, mu, sigma):
    d = x.shape[0]
    argument_of_e = -((x-mu)@np.linalg.inv(sigma)@(x-mu).T)/2
    e = np.exp(argument_of_e)
    num = e.diagonal()
    det = np.linalg.det(sigma)
    denom = ((2*np.pi)**(d/2)) * (np.sqrt(det))
    return num/denom

"""
This function will calculate difference between two Mu vector over all classes
"""
def calculate_error(mu1, mu2, k):
    error = 0
    for i in range(k):
        e = np.sqrt(np.sum((mu1[i]-mu2[i])**2))
        error += e
    return error

"""
This function will calculate the purity of clustering based on the decision
schema given
"""
def calculate_purity(y_true, y_predicted, k):
    
    true_labels = np.unique(y_true)
    predicted_labels = np.unique(y_predicted)
    K = len(true_labels)
    
    sum_overlap = 0
    
    for i in range(k):
        overlap = 0
        Ci = np.argwhere(y_predicted==predicted_labels[i])[...,-1]  
        for j in range(K):
            Tj = np.argwhere(y_true==true_labels[j])[...,-1]
            Ci = set(Ci)
            Tj = set(Tj)
            intersect = len(Ci.intersection(Tj))
            if intersect>overlap:
                overlap = intersect
        sum_overlap += overlap
    return sum_overlap
                
#==============================================================================
    
# MAIN FUNCTION
if __name__ == "__main__":
    
    args = sys.argv
    filename = args[1]
    k = int(args[2])
    eps = float(args[3])
    
    #import file and load data
    X,y = import_and_load_data(filename)
    
    #number of samples
    n = X.shape[0]
    d = X.shape[1]
    
    #encode class labels to int if labels are string
    classNames, y= label_encoder(y)

    
    # INITIALIZATION STEP-------------------->
    Mu = initialize_mu(X,k)     
    Sigma = initialize_sigma(X,k)
    Prior = initialize_prior(k)
    W = np.zeros((k,n))

    
    #repeat expectation and maximization steps
    t = 0
    t_max = 1000
    while(t<t_max):
        
        #copy mu for future comparison
        Mu_copy = np.copy(Mu)
        
        #EXPECTATION STEP----------------------------->
        for i in range(k):
            num = calculate_normal(X, Mu[i], Sigma[i]) * Prior[i]
            denom = 0
            for a in range(k):
                denom += calculate_normal(X, Mu[a], Sigma[a]) * Prior[a]
            #print("in main:",i)
            W[i] = num/denom
               
        #MAXIMIZATION STEP---------------------------->
        for i in range(k):
            weight = W[i]
            weight = weight[:,np.newaxis]
            
            #MU update
            num_sum = np.sum(weight*X, axis=0)
            weight_sum = np.sum(weight, axis=0)
            Mu[i] = num_sum/weight_sum
            
            #Sigma update
            num_sum = 0
            for j in range(n):
                multiply = (X[j]-Mu[i])
                multiply = multiply[:,np.newaxis]
                multiply = multiply@multiply.T
                num_sum += W[i,j]*multiply        
            Sigma[i] = num_sum/weight_sum
            if -(1e-10)<= np.linalg.det(Sigma[i]) <= (1e-10):
                Sigma[i] = Sigma[i]+(1e-3)*np.eye(d)
            
            #Prior update
            Prior[i] = weight_sum / n
        
        #calculate error
        error = calculate_error(Mu_copy, Mu, k)
        
        #check for condition to break
        if error<eps or t>t_max:
            break
        
        #next epoch
        t = t+1
        
    #point to cluster assignment
    y_predicted = np.argmax(W, axis=0)
    labels = np.unique(y_predicted)
    
    #calculate purity score
    purity = (calculate_purity(y, y_predicted, k))/n
    
    #==========================================================================
    #print stuff
    np.set_printoptions(precision=3)
    print("==================================================================")
    print("Means:\n", Mu)
    print("==================================================================")
    print("Covariance Matrices:")
    [print(S) for S in Sigma] 
    print("==================================================================")
    print("Number of Epochs for convergance:", t)
    print("==================================================================")
    print("Cluster Assignments:\n", y_predicted+1)
    print("==================================================================")
    print("Confusion Matrix:\n", confusion_matrix(y, y_predicted))
    print("==================================================================")
    print("Purity:", purity)
    print("==================================================================")
    print("Cluster Size:")
    [print("Class:", classNames[i], ", Size:", len(y_predicted[y_predicted==i])) 
            for i in labels]
    print("==================================================================")
    
    
    