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
#matplotlib inline
import os

# **** Load data ****
path = os.getcwd() + '\ex1data1.txt'
data = pd.read_csv(path, header=None, names=['Population', 'Profit'])

# **** Basic data analysis ****
print(data.head())
print(data.describe())

# **** Define functions *****

def computeCost(X,y,theta):
    inner = np.power((X*theta.T - y),2)
    return np.sum(inner)/(2*len(X))

def gradientDescent(X,y,theta,alpha,iters):
    cost = np.zeros(iters)
    m = len(y);
    
    for i in range(iters):
        h = X*theta.T
        dJ = 1/m * X.T*(h-y);         
        theta = theta - alpha*dJ.T
        cost[i] = computeCost(X,y,theta)
        
    return theta, cost        

def featureNormalize(X):
    mu = np.mean(X,0)
    sigma = np.std(X,0)
    idx = abs(sigma) < 1e-9
    sigma[idx] = 1
    
    X_norm = (X - mu)/sigma
    return X_norm, mu, sigma

# **** Define data vectors ****
data.insert(0,'Ones',1)

cols = data.shape[1]
X = data.iloc[:,0:cols-1]
y = data.iloc[:,cols-1:cols]

X = np.matrix(X.values)
y = np.matrix(y.values)

theta = np.matrix(np.array([0,0]))

print(computeCost(X,y,theta))

iters = 1000
alpha = 0.01
g, cost = gradientDescent(X, y, theta, alpha, iters)

print(g)
print(computeCost(X,y,g))

x = np.linspace(data.Population.min(), data.Population.max(), 100)
f = g[0, 0] + (g[0, 1] * x)

fig, ax = plt.subplots(figsize=(12,8))
ax.plot(x, f, 'r', label='Prediction')
ax.scatter(data.Population, data.Profit, label='Traning Data')
ax.legend(loc=2)
ax.set_xlabel('Population')
ax.set_ylabel('Profit')
ax.set_title('Predicted Profit vs. Population Size')


from sklearn import linear_model  
model = linear_model.LinearRegression()  
model.fit(X, y) 

x = np.array(X[:, 1].A1)  
f = model.predict(X).flatten()

fig, ax = plt.subplots(figsize=(12,8))  
ax.plot(x, f, 'r', label='Prediction')  
ax.scatter(data.Population, data.Profit, label='Traning Data')  
ax.legend(loc=2)  
ax.set_xlabel('Population')  
ax.set_ylabel('Profit')  
ax.set_title('Predicted Profit vs. Population Size') 

fig, ax = plt.subplots(figsize=(12,8))
ax.plot(np.arange(iters), cost, 'r')
ax.set_xlabel('Iterations')
ax.set_ylabel('Cost')
ax.set_title('Error vs. Training Epoch')

theta0_vals = np.linspace(-10,10,100)
theta1_vals = np.linspace(-1,4,101)

X0, X1 = np.meshgrid(theta0_vals,theta1_vals,indexing='ij')
J_cost = np.zeros((100,101))
for i in range(np.shape(theta0_vals)[0]):
    for j in range(np.shape(theta1_vals)[0]):
        theta_val = np.matrix(np.array([theta0_vals[i],theta1_vals[j]]))
        J_cost[i,j] = computeCost(X,y,theta_val)


fig = plt.figure(figsize=(14,6))
ax = fig.add_subplot(1, 2, 1, projection='3d')
ax.plot_surface(X0, X1, J_cost, rstride=10, cstride=10, antialiased=False, cmap = cm.jet)
ax.set_xlabel('theta0')
ax.set_ylabel('theta1')
plt.show()


# **** Load data ****
path = os.getcwd() + '\ex1data2.txt'
data2 = pd.read_csv(path, header=None, names=['Size', 'Bedrooms','Price'])
data2 = (data2 - data2.mean())/data2.std()
data2.insert(0,'Ones',1)

# **** Basic data analysis ****
print(data2.head())
print(data2.describe())

cols = data2.shape[1]
X = data2.iloc[:,0:cols-1]
y = data2.iloc[:,cols-1:cols]
X = np.matrix(X.values)
y = np.matrix(y.values)

theta = np.matrix(np.array([0,0,0]))
 
#Xn, mu, sigma = featureNormalize(X)

iters = 1000
alpha1 = 0.01
alpha2 = 0.03
alpha3 = 0.1
g1, cost1 = gradientDescent(X, y, theta, alpha1, iters)
g2, cost2 = gradientDescent(X, y, theta, alpha2, iters)
g3, cost3 = gradientDescent(X, y, theta, alpha3, iters)

fig, ax = plt.subplots(figsize=(12,8))
ax.plot(np.arange(iters),cost1,'b',label='alpha = 0.01')
ax.plot(np.arange(iters),cost2,'r',label='alpha = 0.03')
ax.plot(np.arange(iters),cost3,'g',label='alpha = 0.1')
legend = ax.legend(loc='upper right', shadow=True)
ax.set_xlabel('Iter')
ax.set_ylabel('Cost')
plt.show()

