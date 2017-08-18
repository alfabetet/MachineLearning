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
from sklearn.preprocessing import PolynomialFeatures

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
    
    theta = np.matrix(theta)
    X = np.matrix(X)
    y = np.matrix(y)
    
    c = np.zeros(iters)
    m = len(y)
    
    for i in range(iters):
        h = sigmoid(X*theta.T)
        dJ = 1/m * X.T*(h-y)        
        theta = theta - alpha*dJ.T
        c[i] = cost(theta,X,y)
        
    return theta, c        

def desicionBoundary(X,theta):
    
    plot_x = np.array([np.min(X[:,1]), np.max(X[:,1])])    
    
    plot_y = -1/theta[2]*(theta[1] * plot_x + theta[0])
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

def costReg(theta, X, y, learningRate):  
    theta = np.matrix(theta)
    X = np.matrix(X)
    y = np.matrix(y)
    first = np.multiply(-y, np.log(sigmoid(X * theta.T)))
    second = np.multiply((1 - y), np.log(1 - sigmoid(X * theta.T)))
    reg = (learningRate / 2 * len(X)) * np.sum(np.power(theta[:,1:theta.shape[1]], 2))
    return np.sum(first - second) / (len(X)) + reg

def gradientReg(theta, X, y, learningRate):  
    theta = np.matrix(theta)
    X = np.matrix(X)
    y = np.matrix(y)

    parameters = int(theta.ravel().shape[1])
    grad = np.zeros(parameters)

    error = sigmoid(X * theta.T) - y

    for i in range(parameters):
        term = np.multiply(error, X[:,i])

        if (i == 0):
            grad[i] = np.sum(term) / len(X)
        else:
            grad[i] = (np.sum(term) / len(X)) + ((learningRate / len(X)) * theta[:,i])

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
X = np.array(X.values)
y = np.array(y.values)
theta = np.array([0,0,0])



c = cost(theta,X,y)

result = opt.fmin_tnc(func=cost, x0=theta, fprime=gradient2, args=(X, y))  
cn = cost(result[0], X, y) 

theta_min = np.matrix(result[0])  
predictions = predict(theta_min, X)  
correct = [1 if ((a == 1 and b == 1) or (a == 0 and b == 0)) else 0 for (a, b) in zip(predictions, y)]  
accuracy = (sum(map(int, correct)) % len(correct))  
print('accuracy = {0}%'.format(accuracy))


plot_x, plot_y = desicionBoundary(X,result[0])

# Plot desicion boundary
fig, ax = plt.subplots(figsize=(12,8))  
ax.scatter(positive['Exam1'], positive['Exam2'], s=50, c='b', marker='o', label='Admitted')  
ax.scatter(negative['Exam1'], negative['Exam2'], s=50, c='r', marker='x', label='Not Admitted')  
ax.plot(plot_x,plot_y)
ax.legend()  
ax.set_xlabel('Exam 1 Score')  
ax.set_ylabel('Exam 2 Score')  


##ex1 = np.arange(np.min(X[:,1]),np.max(X[:,1]))
##ex2 = np.arange(np.min(X[:,2]),np.max(X[:,2]))
##
##theta_min = theta  
##predictions = predict(theta_min, X)  
##correct = [1 if ((a == 1 and b == 1) or (a == 0 and b == 0)) else 0 for (a, b) in zip(predictions, y)]  
##accuracy = (sum(map(int, correct)) % len(correct))  
##print('accuracy = {0}%'.format(accuracy)) 
##
##


# ******************** Regularized *************************
path = os.getcwd() + '\ex2data2.txt'  
data2 = pd.read_csv(path, header=None, names=['Test 1', 'Test 2', 'Accepted'])

positive = data2[data2['Accepted'] > 0.5]
negative = data2[data2['Accepted'] < 0.5]

fig, ax = plt.subplots(figsize=(12,8))  
ax.scatter(positive['Test 1'], positive['Test 2'], s=50, c='b', marker='o', label='Accepted')  
ax.scatter(negative['Test 1'], negative['Test 2'], s=50, c='r', marker='x', label='Rejected')  
ax.legend()  
ax.set_xlabel('Test 1 Score')  
ax.set_ylabel('Test 2 Score')

print(data2.head())

degree = 6 
x1 = data2['Test 1']  
x2 = data2['Test 2']

data2.insert(3, 'Ones', 1)

for i in range(1, degree+1):  
    for j in range(0, i+1):
        print(str(i-j))
        data2['F' + str(i-j) + str(j)] = np.power(x1, i-j) * np.power(x2, j)

data2.drop('Test 1', axis=1, inplace=True)  
data2.drop('Test 2', axis=1, inplace=True)

print(data2.head())

# set X and y (remember from above that we moved the label to column 0)
cols = data2.shape[1]  
X2 = data2.iloc[:,1:cols]  
y2 = data2.iloc[:,0:1]

# convert to numpy arrays and initalize the parameter array theta
X2 = np.array(X2.values)  
y2 = np.array(y2.values)  
theta2 = np.zeros(X2.shape[1])

learningRate = 0

c = costReg(theta2, X2, y2, learningRate)

result2 = opt.fmin_tnc(func=costReg, x0=theta2, fprime=gradientReg, args=(X2, y2, learningRate))  
result2

theta_min = np.matrix(result2[0])  
predictions = predict(theta_min, X2)  
correct = [1 if ((a == 1 and b == 1) or (a == 0 and b == 0)) else 0 for (a, b) in zip(predictions, y2)]  
accuracy = (sum(map(int, correct)) % len(correct))  
print('accuracy = {0}%'.format(accuracy))

plot_x, plot_y = desicionBoundary(X2,result2[0])

# Plot scatter
fig, ax = plt.subplots(figsize=(12,8))  
ax.scatter(positive['Test 1'], positive['Test 2'], s=50, c='b', marker='o', label='Accepted')  
ax.scatter(negative['Test 1'], negative['Test 2'], s=50, c='r', marker='x', label='Rejected')  


# Plot decisionboundary
poly = PolynomialFeatures(degree)
x1_min, x1_max = X2[:,1].min(), X2[:,1].max(),
x2_min, x2_max = X2[:,2].min(), X2[:,2].max(),
xx1, xx2 = np.meshgrid(np.linspace(x1_min, x1_max), np.linspace(x2_min, x2_max))
h = sigmoid(poly.fit_transform(np.c_[xx1.ravel(), xx2.ravel()]).dot(result2[0]))
h = h.reshape(xx1.shape)
ax.contour(xx1, xx2, h, [0.5], linewidths=1, colors='g');       
ax.set_title('Train accuracy {}% with Lambda = {}'.format(np.round(accuracy, decimals=2), C))
    
ax.legend()  
ax.set_xlabel('Test 1 Score')  
ax.set_ylabel('Test 2 Score')

for i in range(0,5):
    print(i)