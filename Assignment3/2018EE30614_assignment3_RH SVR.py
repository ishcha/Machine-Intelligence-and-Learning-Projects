#!/usr/bin/env python
# coding: utf-8

# # RH SVR on the Boston Housing Dataset

# ## Importing the libraries

# In[38]:


import numpy as np
import math
import pandas as pd
from cvxopt import matrix, solvers
solvers.options['show_progress'] = False


# ## Fetching the data

# In[6]:


data = pd.read_csv('housing.csv', header = None)


# In[7]:


print(data.shape)


# In[8]:


data.columns = ['CRIM',
            'ZN',
            'INDUS',
            'CHAS', 
            'NOX',
            'RM', 
            'AGE',
            'DIS',
            'RAD',
            'TAX',
            'PTRATIO',
            'B',
            'LSTAT',
            'MEDV']


# In[9]:


data.head()


# ## Standardizing the data

# In[10]:


X = data.drop('MEDV', axis = 1).copy()
X.head()


# In[11]:


y = data['MEDV'].copy()
y.head()


# In[12]:


data.shape


# In[13]:


X_mean = np.mean(X)
print(X_mean.shape)


# In[14]:


X_std = np.std(X)
print(X_std.shape)


# In[15]:


X_norm = (X - X_mean)/X_std
print(X_norm.shape)
X_norm.head()


# In[16]:


X_norm = X_norm.to_numpy()
y = y.to_numpy()


# In[17]:


data = np.column_stack((X_norm, y))
data.shape


# In[18]:


data_test = data[0: 125]
data_train = data[125:]
print(data_test, data_train)


# ## Function to formulate the optimization problem and solve it using the solvers

# In[36]:


def optimize(y, K, epsilon, D):
    m = K + np.outer(y, y.T)
    n = len(y)
    i = np.vstack((np.eye(n), -1*np.eye(n)))
    P = matrix(np.dot(np.dot(i, m), i.T))
    q = matrix(2*epsilon*np.dot(i, y))
    e = np.ones((1, len(y)))
    O = np.zeros((1, len(y)))
    I = np.identity(len(y))
    G = matrix(np.vstack((-1*np.eye(2*n), np.eye(2*n))))
    h = matrix(np.vstack((np.zeros((2*n, 1)), D*np.ones((2*n, 1)))))
    A = matrix(np.vstack((np.hstack((e, O)), np.hstack((O, e)))))
    b = matrix(np.ones((2, 1)))
    
    sol = solvers.qp(P,q,G,h,A,b)
    u_hat = sol['x'][0:len(y)]
    v_hat = sol['x'][len(y):]
    return u_hat, v_hat, sol['primal objective']    
    


# ## Computing the regression function

# In[21]:


def fun(y, epsilon, K, D):
    u_hat, v_hat, prim_obj = optimize(y, K, epsilon, D)
    delta_hat = np.dot((u_hat - v_hat).T, y) + 2*epsilon
    u_bar = u_hat/delta_hat
    v_bar = v_hat/delta_hat
    b_bar = np.dot((u_hat - v_hat).T, np.dot(K, (u_bar + v_bar)))/2 + np.dot((u_hat + v_hat).T, y)/2
    epsilon_hat = -1*np.dot((u_hat - v_hat).T, np.dot(K, (u_bar - v_bar)))/2 + np.dot((v_hat - u_hat).T, y)/2
    return (v_bar - u_bar), b_bar, epsilon_hat, prim_obj
    


# ## Kernel functions

# In[75]:


def compute_RBF_kernel_matrix(X1, X2, gamma):
    K = np.zeros((X1.shape[0], X2.shape[0]))  
    for i1 in range(X1.shape[0]):
        for j1 in range(X2.shape[0]):
            u = X1[i1]-X2[j1]
            #print(u)
            K[i1][j1] = np.exp(-1*gamma*np.dot(u, u.T))
    return K


# In[23]:


compute_RBF_kernel_matrix(data_test[:, :-1], data_test[:, :-1], 1)


# ## Using cvxopt to find the regression function

# ### Setting the hyper-parameters: 

# In[24]:


epsilon = 1
gamma = 0.0001
D = 10


# ## Creating the test set

# In[70]:


np.random.shuffle(data)
data_test = data[0:100]
data_train = data[100:]


# In[26]:


print(data_test.shape, data_train.shape)


# ## To find the R2 score

# In[58]:


def evaluate_svr(X_train, y_train, y_test, X_test, A, b_bar, epsilon_hat, gamma):
    y_hat = (compute_RBF_kernel_matrix(X_train, X_test, gamma).T).dot(A) + b_bar
    Y = np.reshape(y_test, (len(y_test), 1))
    e1 = np.subtract(Y,y_hat)
    E1 = np.dot(e1.T, e1)[0, 0]
    me = np.mean(Y)
    e2 = np.subtract(Y,me)
    E2 = np.dot(e2.T, e2)[0, 0]
    r2 = 1-E1/E2
    
    y_hat1 = (compute_RBF_kernel_matrix(X_train, X_train, gamma).T).dot(A) + b_bar
    Y1 = np.reshape(y_train, (len(y_train), 1))
    e11 = np.subtract(Y1,y_hat1)
    E11 = np.dot(e11.T, e11)[0, 0]
    me1 = np.mean(Y1)
    e21 = np.subtract(Y1,me1)
    E21 = np.dot(e21.T, e21)[0, 0]
    r21 = 1-E11/E21

    return r2, r21, y_hat, y_hat1


# In[76]:


def svr(data_train, data_test, epsilon, D, gamma):
    X_train = data_train[:, :-1]
    y_train = data_train[:, -1]
    X_test = data_test[:, :-1]
    y_test = data_test[:, -1]
    K = compute_RBF_kernel_matrix(X_train, X_train, gamma)
    A, b_bar, epsilon_hat, prim_obj = fun(y_train, epsilon, K, D)
    r2, r21, y_hat, y_hat1 = evaluate_svr(X_train, y_train, y_test, X_test, A, b_bar, epsilon_hat, gamma)
    return r2, r21


# ## Implementation:

# In[74]:


s = svr(data_train, data_test, 0.001, 10, 0.0001)
print('R2 Score on test data: {0:.3f}, R2 Score on training data: {1:.3f}'.format(s[0], s[1]))

