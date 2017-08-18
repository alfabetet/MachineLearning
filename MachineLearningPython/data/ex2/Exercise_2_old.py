# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import scipy.optimize as opt  
#matplotlib inline
import os

# **** Function definitions ****
def sigmoid(z):
    return 1/(1+np.exp(-z))

def cost(theta, X, y):  
    theta = np.matrix(theta)
    X = np.matrix(X)
    y = np.matrix(y)
    first = np.multiply(-y, np.log(sigmoid(X * theta.T)))
    second = np.multiply((1 - y), np.log(1 - sigmoid(X * theta.T)))
    return np.sum(first - second) / (len(X))

def gradientDescent(X,y,theta,alpha,iters):
    c = np.zeros(iters)
    m = len(y);
    
    for i in range(iters):
        h = sigmoid(X*theta.T)
        dJ = 1/m * X.T*(h-y)        
        theta = theta - alpha*dJ.T
        c[i] = cost(X,y,theta)
        
    return theta, c        

def desicionBoundary(X,theta):
    
    plot_x = np.array([np.min(X[:,1]), np.max(X[:,1])])    
    
    plot_y = -1/theta[0,2]*(theta[0,1] * plot_x + theta[0,0])
    return plot_x, plot_y

def predict(theta, X):  
    probability = sigmoid(X * theta.T)
    return [1 if x >= 0.5 else 0 for x in probability]

def gradient(theta, X, y):  
    theta = np.matrix(theta)
    m = len(y)
    X = np.matrix(X)
    y = np.matrix(y)
#
#    parameters = int(theta.ravel().shape[1])
#    grad = np.zeros(parameters)
#
#    error = sigmoid(X * theta.T) - y

#    for i in range(parameters):
#        term = np.multiply(error, X[:,i])
#        grad[i] = np.sum(term) / len(X)

    h = sigmoid(X*theta.T)
    grad = 1/m * X.T*(h-y) 

    return grad.T

def gradient2(theta, X, y):  
    theta = np.matrix(theta)
    X = np.matrix(X)
    y = np.matrix(y)

    parameters = int(theta.ravel().shape[1])
    grad = np.zeros(parameters)

    error = sigmoid(X * theta.T) - y

    for i in range(parameters):
        term = np.multiply(error, X[:,i])
        grad[i] = np.sum(term) / len(X)

    return grad

# **** Load data ****
path = os.getcwd() + '\ex2data1.txt'
data = pd.read_csv(path, header=None, names=['Exam1', 'Exam2','Admitted'])

# **** Basic data analysis ****
print(data.head())
print(data.describe())

# **** Plot data ****
positive = data[data['Admitted'] > 0.5]
negative = data[data['Admitted'] < 0.5]

fig, ax = plt.subplots(figsize=(12,8))  
ax.scatter(positive['Exam1'], positive['Exam2'], s=50, c='b', marker='o', label='Admitted')  
ax.scatter(negative['Exam1'], negative['Exam2'], s=50, c='r', marker='x', label='Not Admitted')  
ax.legend()  
ax.set_xlabel('Exam 1 Score')  
ax.set_ylabel('Exam 2 Score')  

nums = np.arange(-10, 10, step=0.5)

fig, ax = plt.subplots(figsize=(12,8))  
ax.plot(nums, sigmoid(nums), c='r',marker='o')

nums = np.arange(0.1, 1, step=0.01)

fig, ax = plt.subplots(figsize=(12,8))  
ax.plot(nums, -np.log(nums), c='r',marker='o')
ax.plot(nums, -np.log(1-nums), c='r',marker='x')

# add a ones column - this makes the matrix multiplication work out easier
data.insert(0, 'Ones', 1)

# set X (training data) and y (target variable)
cols = data.shape[1]  
X = data.iloc[:,0:cols-1]  
y = data.iloc[:,cols-1:cols]

# convert to numpy arrays and initalize the parameter array theta
X = np.matrix(np.array(X.values))  
y = np.matrix(np.array(y.values))
theta = np.matrix(np.array([0,0,0]))



c = cost(X,y,theta)

# gradient descent
alpha = 0.001
iters = 40
theta, c = gradientDescent(X, y, theta, alpha, iters)

fig, ax = plt.subplots(figsize=(12,8))
ax.plot(np.arange(iters), c, 'r')
ax.set_xlabel('Iterations')
ax.set_ylabel('Cost')
ax.set_title('Error vs. Training Epoch')
#
#plot_x, plot_y = desicionBoundary(X,theta)
#
## Plot desicion boundary
#fig, ax = plt.subplots(figsize=(12,8))  
#ax.scatter(positive['Exam1'], positive['Exam2'], s=50, c='b', marker='o', label='Admitted')  
#ax.scatter(negative['Exam1'], negative['Exam2'], s=50, c='r', marker='x', label='Not Admitted')  
#ax.plot(plot_x,plot_y)
#ax.legend()  
#ax.set_xlabel('Exam 1 Score')  
#ax.set_ylabel('Exam 2 Score')  
#
#
#ex1 = np.arange(np.min(X[:,1]),np.max(X[:,1]))
#ex2 = np.arange(np.min(X[:,2]),np.max(X[:,2]))
#
#theta_min = theta  
#predictions = predict(theta_min, X)  
#correct = [1 if ((a == 1 and b == 1) or (a == 0 and b == 0)) else 0 for (a, b) in zip(predictions, y)]  
#accuracy = (sum(map(int, correct)) % len(correct))  
#print('accuracy = {0}%'.format(accuracy)) 
#
#
#result = opt.fmin_tnc(func=cost, x0=theta, fprime=gradient2, args=(X, y))  
#cn = cost(result[0], X, y) 
#
#theta_min = result[0]  
#predictions = predict(theta_min, X)  
#correct = [1 if ((a == 1 and b == 1) or (a == 0 and b == 0)) else 0 for (a, b) in zip(predictions, y)]  
#accuracy = (sum(map(int, correct)) % len(correct))  
#print('accuracy = {0}%'.format(accuracy)) 