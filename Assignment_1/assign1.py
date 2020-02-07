# -*- coding: utf-8 -*-
"""
Created on Sun Sep 15 15:26:11 2019

@author: nafisneehal
"""

#import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys


def plot_data(dataset, attr1, attr2, title):
    x = dataset[:,attr1]
    y = dataset[:,attr2]
    plt.scatter(x,y)
    z = np.polyfit(x, y, 1)
    p = np.poly1d(z)
    plt.plot(x,p(x),"r--")
    plt.xlabel('Attribute '+str(attr1))
    plt.ylabel('Attribute '+str(attr2))
    plt.title(title)
    plt.show()
    
def calculate_mean(dataset, axis):
    return np.mean(dataset, axis=axis)

def calculate_total_variance(dataset, mean_vector):
    centered_matrix = dataset - mean_vector #centerized vector
    squared_centered_matrix = centered_matrix**2
    sample_variance = np.sum(squared_centered_matrix, axis=1) #sqrt{(x1-y1)**2 + (x2-y2)**2 + .. + (xn-yn)**2} between two vectors with n components
    total_variance = np.sum(sample_variance)/samples
    return centered_matrix, total_variance

def calculate_covariance_matrix(centered_matrix, samples):
    return (1/samples)*(centered_matrix.T@centered_matrix) #6x6 cov matrix

def calculate_magnitude(vector):
    return np.sqrt(vector.dot(vector))

def calculate_distance(a,b):
    return np.sqrt(np.sum((a-b)**2))
    
def calculate_inner_product(centered_matrix, samples):
    return (centered_matrix.T@centered_matrix) / samples 

def calculate_outer_product(centered_matrix, samples):
    outer_product = np.zeros((6,6))
    for i in range(centered_matrix.shape[0]):
        row_vector = centered_matrix[i][:, np.newaxis]
        row_matrix = row_vector@row_vector.T
        outer_product = outer_product + row_matrix
    outer_product = outer_product/samples
    return outer_product

def calculate_corr_pairs(attributes, centered_matrix ):
    correlation_vector = np.zeros((attributes, attributes))
    distance_from_zero = 100000
    least_correlation = None
    
    for i in range(centered_matrix.shape[1]):
        for j in range(centered_matrix.shape[1]):
            if i<j:
                correlation_vector[i,j] = (centered_matrix[:,i].T@centered_matrix[:,j])
                correlation_vector[i,j] = correlation_vector[i,j] / (np.sqrt(centered_matrix[:,i].dot(centered_matrix[:,i]))*(np.sqrt(centered_matrix[:,j].dot(centered_matrix[:,j]))))
                #for calculating the least correlated, abs(point-0), the closer the correlation is to 0, the lesser the correlation.
                if(abs(correlation_vector[i,j]-0)<distance_from_zero):
                    distance_from_zero = abs(correlation_vector[i,j]-0)
                    least_correlation = np.argwhere(correlation_vector==correlation_vector[i,j])
                       
    #most correlated and anti-correlated
    max_correlation = np.argwhere(correlation_vector==correlation_vector.max())
    max_anti_correlation = np.argwhere(correlation_vector==correlation_vector.min())
    
    return correlation_vector, max_correlation, max_anti_correlation, least_correlation
    

if __name__ == '__main__':
    
    arguments = sys.argv
    filename = arguments[1]
    epsilon = arguments[2]
    
    
    #import datafile
    #df = pd.read_csv(filename, sep='\t', header=None)
    df = pd.read_csv(filename, sep='\t', header=None)
    dataset = df.to_numpy()
    
    #extract data dimension
    samples = dataset.shape[0]
    attributes = dataset.shape[1]
    
    '''Part 1'''
    #### Problem (A) ####
    
    #calculate mean vector
    mean_vector = calculate_mean(dataset, 0)
    print ("Mean Vector:", mean_vector)
    #calculate total variance
    centered_matrix, total_variance = calculate_total_variance(dataset, mean_vector)
    
    
    
    #### Problem (B) ####

    #calculate inner product
    inner_product = calculate_inner_product(centered_matrix, samples)
    #calculate outer product
    outer_product = calculate_outer_product(centered_matrix, samples)
    
    
    
    #### Problem (C) ####
    
    correlation_vector, max_correlation, max_anti_correlation, least_correlation = calculate_corr_pairs(attributes, centered_matrix)
    #plotting maximum correlation + general trendline 
    plot_data(dataset, max_correlation[0][0], max_correlation[0][1],"Maximum Correlation" )
    #plotting max_anti_correlation + general trendline
    plot_data(dataset, max_anti_correlation[0][0], max_anti_correlation[0][1], "Maximum Anti-Correlation")
    #plotting least_correlation + general trendline 
    plot_data(dataset, least_correlation[0][0], least_correlation[0][1], "Least Correlation" )


    '''Part 2'''
    
    #calculate covariance matrix
    distance = 10000
    covariance_matrix = calculate_covariance_matrix(centered_matrix, samples) #6x6
    eigenvector = np.random.randint(1,7,size=(6,2))
    
    while(distance>epsilon):
        
        print(eigenvector[:,0],eigenvector[:,1])
        
        #store old eigenvector
        old_eigenvector = eigenvector
        
        #orthogonalize b
        eigenvector[:,1] = eigenvector[:,1] - ((eigenvector[:,1].T@eigenvector[:,0]) / eigenvector[:,0].T@eigenvector[:,0])*eigenvector[:,0]
        
        #normalize old vector with new norm of b
        
        eigenvector[:,0] = eigenvector[:,0] / calculate_magnitude(eigenvector[:,0])
        eigenvector[:,1] = eigenvector[:,1] / calculate_magnitude(eigenvector[:,1])
        
        #eigenvector update
        eigenvector = covariance_matrix@eigenvector
        
        #normalize new vector and assign a,b
        eigenvector[:,0] = eigenvector[:,0] / calculate_magnitude(eigenvector[:,0])
        eigenvector[:,1] = eigenvector[:,1] / calculate_magnitude(eigenvector[:,1])
        
        #compare old and new eigenvector
        distance = calculate_distance(old_eigenvector, eigenvector)
        
    
        
        
    
    
    
    
    
    
    
    
    
    
    
    
    
    