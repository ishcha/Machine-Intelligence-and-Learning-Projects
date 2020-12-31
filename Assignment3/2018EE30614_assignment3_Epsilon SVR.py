#!/usr/bin/env python
# coding: utf-8

# # Epsilon SVR using the Boston Dataset: Implementation using CVXOPT and SCIKIT Learn

# ## Data imports and processing

# In[61]:


import numpy as np
import pandas as pd
from cvxopt import matrix, solvers
from numpy.linalg import matrix_rank
import math
solvers.options['show_progress'] = False
import matplotlib.pyplot as plt


# In[2]:


data = pd.read_csv('housing.csv', header = None)


# In[3]:


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


# In[4]:


data.head()


# In[5]:


data = data.sample(frac = 1).reset_index(drop = True)


# In[6]:


data.head()


# ### Standardizing the data

# In[7]:


X = data.drop('MEDV', axis = 1).copy()
X.head()


# In[8]:


y = data['MEDV'].copy()
y.head()


# In[9]:


data.shape


# In[10]:


X_mean = np.mean(X)
print(X_mean.shape)


# In[11]:


X_std = np.std(X)
print(X_std.shape)


# In[12]:


X_norm = (X - X_mean)/X_std
print(X_norm.shape)
X_norm.head()


# In[13]:


data_norm = np.column_stack((X_norm, y))
print(data_norm.shape, type(data_norm))
data_norm[0]


# ### Splitting training and test data

# In[14]:


np.random.shuffle(data_norm)
data_test = data_norm[0:100]
data_train = data_norm[100:]
print(data_test.shape, data_train.shape)


# ## Kernel functions

# In[41]:


def Compute_kernel(X1, X2, g, k):
    K = np.zeros((X1.shape[0], X2.shape[0]))
    if k == 'rbf':
        for i in range(X1.shape[0]):
            for j in range(X2.shape[0]):
                u = X1[i]-X2[j]
                K[i][j] = np.exp(-1*g*np.dot(u, u.T))
    elif k == 'poly':
        for i in range(X1.shape[0]):
            for j in range(X2.shape[0]):
                K[i][j] = (1 + np.dot(X1[i], X2[j].T))**g
    elif k == 'linear':
        for i in range(X1.shape[0]):
            for j in range(X2.shape[0]):
                K[i][j] = np.dot(X1[i], X2[j].T)
    
    return K


# In[31]:


Compute_kernel(data_train[:, :-1], data_train[:, :-1], 0.1, 'sigmoid')


# ## CVXOPT implementation

# In[16]:


class svr_cvxopt:
    def __init__(self, kernel):
        self.k = kernel


# In[17]:


class svr_cvxopt(svr_cvxopt):
    def optimize(self, X, y, epsilon, C, g):
        n = X.shape[0]
        I = np.vstack((np.eye(n), -1*np.eye(n)))
        K = Compute_kernel(X, X, g, self.k)
        P1 = np.dot(I, np.dot(K, I.T))
        P = matrix(P1)
        
        e = epsilon*np.ones((1, n))
        y1 = e - y.T
        y2 = e + y.T
        q1 = np.hstack((y1, y2)).T
        q = matrix(q1)
        
        I1 = np.eye(2*n)
        G1 = np.vstack((I1, -1*I1))
        G = matrix(G1)
        
        c = C*np.ones((2*n, 1))
        o = np.zeros((2*n, 1))
        h1 = np.vstack((c, o))
        h = matrix(h1)
        
        one = np.ones((1, n))
        A1 = np.hstack((one, -1*one))
        A = matrix(A1)
        
        b1 = np.array([0.0])
        b = matrix(b1)
 
        sol = solvers.qp(P, q, G, h, A, b)
        a = sol['x'][0:n]
        a_hat = sol['x'][n:]
        return a, a_hat, sol['primal objective']
    
    def reg_fun(self, X, y, epsilon, C, g):
        a, a_hat, prim_obj = self.optimize(X, y, epsilon, C, g)
        a = np.array(a)
        a_hat = np.array(a_hat)
        W = a - a_hat
        b = 0.0
        t = 0
        for v in range(len(a)):
            if ((a[v] > 1e-3) and (a[v] < C)) or ((a_hat[v] > 1e-3) and (a_hat[v] < C)):
                V = np.reshape(X[v, :], (1, len(X[v, :])))
                b += y[v] - epsilon -np.dot(Compute_kernel(V, X, g, self.k), (a-a_hat))
                t += 1
        b /= t
        return W , b            


# In[120]:


class svr_cvxopt(svr_cvxopt):
    def evaluate(self, X_train, y_train, X_test, y_test, epsilon, C, g):
        W, b = self.reg_fun(X_train, y_train, epsilon, C, g)
        y_hat = np.dot(Compute_kernel(X_test, X_train, g, self.k), W) + b
        
        Y = np.reshape(y_test, (len(y_test), 1))
        error = np.absolute(y_hat - Y)
        loss = error - epsilon
        loss[loss < 0] = 0
        return loss, y_hat


# In[57]:


class svr_cvxopt(svr_cvxopt):
    def r2score(self, X_train, y_train, X_test, y_test, epsilon, C, g):
        W, b = self.reg_fun(X_train, y_train, epsilon, C, g)
        y_hat = np.dot(Compute_kernel(X_test, X_train, g, self.k), W) + b
        y_hat1 = np.dot(Compute_kernel(X_train, X_train, g, self.k), W) + b
        Y = np.reshape(y_test, (len(y_test), 1))
        e1 = np.subtract(Y,y_hat)
        E1 = np.dot(e1.T, e1)[0, 0]
        me = np.mean(Y)
        e2 = np.subtract(Y,me)
        E2 = np.dot(e2.T, e2)[0, 0]
        r2 = 1-E1/E2
        
        Y1 = np.reshape(y_train, (len(y_train), 1))
        e11 = np.subtract(Y1,y_hat1)
        E11 = np.dot(e11.T, e11)[0, 0]
        me1 = np.mean(Y1)
        e21 = np.subtract(Y1,me1)
        E21 = np.dot(e21.T, e21)[0, 0]
        r21 = 1-E11/E21
        return r2, r21, y_hat, y_hat1


# In[20]:


class svr_cvxopt(svr_cvxopt):
    def mse(self, X_train, y_train, X_test, y_test, epsilon, C, g):
        W, b = self.reg_fun(X_train, y_train, epsilon, C, g)
        y_hat = np.dot(Compute_kernel(X_test, X_train, g, self.k), W) + b
        Y = np.reshape(y_test, (len(y_test), 1))
        e1 = np.subtract(Y,y_hat)
        E1 = np.dot(e1.T, e1)/X_test.shape[0]
        
        y_hat1 = np.dot(Compute_kernel(X_train, X_train, g, self.k), W) + b
        Y1 = np.reshape(y_train, (len(y_train), 1))
        e11 = np.subtract(Y1,y_hat1)
        E11 = np.dot(e11.T, e11)/X_train.shape[0]
        return E1, E11, y_hat, y_hat1


# In[21]:


class svr_cvxopt(svr_cvxopt):
    def fit(self, X_train, y_train, X_test, y_test, epsilon, C, g):
        W, b = self.reg_fun(X_train, y_train, epsilon, C, g)
        y_hat = np.dot(Compute_kernel(X_test, X_train, g, self.k), W) + b
        return y_hat


# ### Applying CVXOPT implementation

# In[119]:


model = svr_cvxopt('rbf')
r2_test, r2_train, y_hat_test, y_hat_train = model.r2score(data_train[:, :-1], data_train[:, -1], data_test[:, :-1], data_test[:, -1], 0.001, 10, 0.0001)
print('R2Score(Test): ', r2_test)
print('R2Score(Train): ', r2_train)
# print(y_hat_test)
# print(y_hat_train)
#print(y_hat_test.shape)


# ## Grid Search Cross Validation: Self implementation

# In[48]:


def GridCV(params, x, y, k1 = 5):
    n = x.shape[0]
    c = params['C']
    e = params['epsilon']
    ker = params['kernel']
    g = params['gamma']
    mini = 10000000.0
    bs = math.floor(x.shape[0]/k1)
    rem = x.shape[0]%k1
    BS = [bs+1]*rem + [bs]*(k1-rem)
    pars = dict.fromkeys(['C', 'epsilon', 'kernel', 'gamma'])
    for i in range(len(ker)):
        model = svr_cvxopt(ker[i])
        for j in range(len(e)):
            for k in range(len(c)):
                for l in range(len(g)):
                    h = 0
                    losses = np.zeros((1, x.shape[0]))
                    for y1 in range(k1):
                        m = BS[y1]
                        ax = x[h:h+m]
                        ay = y[h:h+m]
                        rx = np.concatenate((x[0:h], x[h+m:n]), axis = 0)
                        ry = np.concatenate((y[0:h], y[h+m:n]), axis = 0)                
                        y_hat = model.fit(rx, ry, ax, ay, e[j], c[k], g[l])
                        for r in range(len(ax)):
                            losses[0, h+r] = ((ay[r]-y_hat[r])**2)
                        h += BS[y1] 
                        
                    s = np.sum(losses)/x.shape[0]
                    if (s < mini):
                        mini = s
                        pars['C'] = c[k]
                        pars['epsilon'] = e[j]
                        pars['kernel'] = ker[i]
                        pars['gamma'] = g[l]
    return pars


# ### Application of Grid Search Cross Validation

# In[112]:


param_grid = {
                'C': np.linspace(0.1, 10, 2),
                'epsilon': [0.1, 0.5, 1],
                'kernel': ['rbf'],
                'gamma': [0.001, 0.1]
            }


ans = GridCV(param_grid, data_train[:, :-1], data_train[:, -1], 5)
print(ans)


# ## Plotting the influence of hyper parameters and kernel functions over the R2 scores

# In[87]:


r_test = []
p_test = []
l_test = []
r_train = []

model1 = svr_cvxopt('rbf')
epsilon = 0.1
gamma = 0.1
C_list = [0.1, 1, 5, 10, 50, 100, 500, 1000]
for c in range(len(C_list)):
    r2_test, r2_train, y_hat_test, y_hat_train = model1.r2score(data_train[:, :-1], data_train[:, -1], data_test[:, :-1], data_test[:, -1], epsilon, C_list[c], gamma)
    r_test.append(r2_test)
    r_train.append(r2_train)
    
model2 = svr_cvxopt('poly')
epsilon = 0.1
gamma = 2
for c in range(len(C_list)):
    r2_test, r2_train, y_hat_test, y_hat_train = model2.r2score(data_train[:, :-1], data_train[:, -1], data_test[:, :-1], data_test[:, -1], epsilon, C_list[c], gamma)
    p_test.append(r2_test)
    
model3 = svr_cvxopt('linear')
epsilon = 0.1
for c in range(len(C_list)):
    r2_test, r2_train, y_hat_test, y_hat_train = model3.r2score(data_train[:, :-1], data_train[:, -1], data_test[:, :-1], data_test[:, -1], epsilon, C_list[c], gamma)
    l_test.append(r2_test)

plt.plot(C_list, l_test, C_list, r_test, C_list, p_test ) 

plt.xlabel('Variation in C with epsilon = 0.1(for RBF: gamma = 0.1; for polynomial: gamma = 2)')
plt.ylabel('R2 Score of the test data set (100 samples)')

plt.title('Variation of R2 score of test set with C for different kernels')

plt.legend(['Linear kernel', 'RBF kernel', 'Polynomial kernel' ])
plt.show()


# In[88]:


plt.plot(C_list, r_test)
plt.plot(C_list, r_train)

plt.xlabel('Variation in C with epsilon = 0.1, for RBF: gamma = 0.1')
plt.ylabel('R2 scores')

plt.title('Observing overfitting and underfitting of the model for RBF kernel with the variation of the hyper parameter C')
plt.legend(['R2 score for test data', 'R2 score for training data'])
plt.show()


# In[85]:


r_test1 = []
p_test1 = []
l_test1 = []
r_train1 = []

model1 = svr_cvxopt('rbf')
C = 10
gamma = 0.1
epsilon_list = [0.0001, 0.001,0.005, 0.01, 0.05, 0.1, 0.5, 1, 1.5, 2, 5]
for c in range(len(epsilon_list)):
    r2_test, r2_train, y_hat_test, y_hat_train = model1.r2score(data_train[:, :-1], data_train[:, -1], data_test[:, :-1], data_test[:, -1], epsilon_list[c], C, gamma)
    r_test1.append(r2_test)
    r_train1.append(r2_train)
    
model2 = svr_cvxopt('poly')
gamma = 2
for c in range(len(epsilon_list)):
    r2_test, r2_train, y_hat_test, y_hat_train = model2.r2score(data_train[:, :-1], data_train[:, -1], data_test[:, :-1], data_test[:, -1], epsilon_list[c], C, gamma)
    p_test1.append(r2_test)
    
model3 = svr_cvxopt('linear')
for c in range(len(epsilon_list)):
    r2_test, r2_train, y_hat_test, y_hat_train = model3.r2score(data_train[:, :-1], data_train[:, -1], data_test[:, :-1], data_test[:, -1], epsilon_list[c], C, gamma)
    l_test1.append(r2_test)


plt.plot(epsilon_list, r_test1, epsilon_list, l_test1, epsilon_list, p_test1 ) 

plt.xlabel('Variation in epsilon with C = 10(for RBF: gamma = 0.1; for polynomial: gamma = 2)')
plt.ylabel('R2 Score of the test data set (100 samples)')

plt.title('Variation of R2 score of test set with Epsilon for different kernels')

plt.legend(['RBF kernel', 'Linear kernel', 'Polynomial kernel' ])
plt.show()


# In[86]:


plt.plot(epsilon_list, r_test1)
plt.plot(epsilon_list, r_train1)

plt.xlabel('Variation in Epsilon with C = 10, for RBF: gamma = 0.1')
plt.ylabel('R2 scores')

plt.title('Observing overfitting and underfitting of the model for RBF kernel with the variation of the hyper parameter Epsilon')
plt.legend(['R2 score for test data', 'R2 score for training data'])
plt.show()


# In[90]:


r_test2 = []
r_train2 = []

model1 = svr_cvxopt('rbf')
C = 10
gamma_list = [0.0001, 0.001,0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10]
epsilon = 0.1
for c in range(len(gamma_list)):
    r2_test, r2_train, y_hat_test, y_hat_train = model1.r2score(data_train[:, :-1], data_train[:, -1], data_test[:, :-1], data_test[:, -1], epsilon, C, gamma_list[c])
    r_test2.append(r2_test)
    r_train2.append(r2_train)
    

plt.plot(gamma_list, r_test2)
plt.plot(gamma_list, r_train2)

plt.xlabel('Variation in Gamma with C = 10 and epsilon = 0.1 for RBF')
plt.ylabel('R2 scores')

plt.title('Observing overfitting and underfitting of the model for RBF kernel with the variation of the hyper parameter Gamma')
plt.legend(['R2 score for test data', 'R2 score for training data'])
plt.show()


# In[98]:


p_test2 = []
p_train2 = []

model2 = svr_cvxopt('poly')
C = 10
gamma_list = [1, 2, 3]
epsilon = 0.1
for c in range(len(gamma_list)):
    r2_test, r2_train, y_hat_test, y_hat_train = model2.r2score(data_train[:, :-1], data_train[:, -1], data_test[:, :-1], data_test[:, -1], epsilon, C, gamma_list[c])
    p_test2.append(r2_test)
    p_train2.append(r2_train)
    

plt.plot(gamma_list, p_test2)
plt.plot(gamma_list, p_train2)

plt.xlabel('Variation in Gamma with C = 10 and epsilon = 0.1 for Polynomial kernel')
plt.ylabel('R2 scores')

plt.title('Observing overfitting and underfitting of the model for Polynomial kernel with the variation of the hyper parameter Gamma')
plt.legend(['R2 score for test data', 'R2 score for training data'])
plt.show()


# ## Using sklearn library

# In[104]:


from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error


# In[105]:


param_grid = [
    {
        'C': np.linspace(0.1, 100, 10),
        'epsilon': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10],
        'kernel': ['rbf'],
        'gamma': [0.0001, 0.001, 0.005, 0.1, 1, 2, 3, 5]
    }
]

reg = GridSearchCV(
    SVR(), 
    param_grid,
    cv = 5,
    scoring = 'neg_mean_squared_error',
    verbose = 0
)

reg.fit(data_train[:, :-1], data_train[:, -1])
reg.best_params_


# In[110]:


model_reg = SVR(kernel = 'rbf', gamma = 0.005, C = 100, epsilon = 1)
model_reg.fit(data_train[:,:-1], data_train[:, -1])
p = model_reg.predict(data_test[:, :-1])
print(model_reg.score(data_train[:, :-1], data_train[:, -1]))
print(model_reg.score(data_test[:, :-1], data_test[:, -1]))
mean_squared_error(data_test[:, -1], p)


# In[111]:


model_reg.support_vectors_.shape


# ### Forming Bar graph to compare the 2 implementations

# In[117]:


model1 = svr_cvxopt('rbf')
model2 = svr_cvxopt('poly')
model3 = svr_cvxopt('linear')

C = 10
epsilon = 0.1
gamma1 = 0.1
gamma2 = 2

fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
X = np.arange(3)
labels = ['RBF', 'Poly', 'Linear']
ax.set_ylabel('R2 Scores')
ax.set_title('R2 Scores by CVXOPT and SKLEARN')
ax.set_xticks(X)
ax.set_xticklabels(labels)

model_sk1 = SVR(kernel = 'rbf', gamma = gamma1, C = C, epsilon = epsilon)
model_sk2 = SVR(kernel = 'poly', degree = 2, C = C, epsilon = epsilon)
model_sk3 = SVR(kernel = 'linear', C = C, epsilon = epsilon)

model_sk1.fit(data_train[:,:-1], data_train[:, -1])
model_sk2.fit(data_train[:,:-1], data_train[:, -1])
model_sk3.fit(data_train[:,:-1], data_train[:, -1])

r2 = []
r2.append(model_sk1.score(data_test[:, :-1], data_test[:, -1]))
r2.append(model_sk2.score(data_test[:, :-1], data_test[:, -1]))
r2.append(model_sk3.score(data_test[:, :-1], data_test[:, -1]))



ax.bar(X, r2, color = 'b', width = 0.25, label = 'sklearn')

r2_test1, _, _, _ = model1.r2score(data_train[:, :-1], data_train[:, -1], data_test[:, :-1], data_test[:, -1], epsilon, C, gamma1)
r2_test2, _, _, _ = model2.r2score(data_train[:, :-1], data_train[:, -1], data_test[:, :-1], data_test[:, -1], epsilon, C, gamma2)
r2_test3, _, _, _ = model3.r2score(data_train[:, :-1], data_train[:, -1], data_test[:, :-1], data_test[:, -1], epsilon, C, gamma1)

R2 = []
R2.append(r2_test1)
R2.append(r2_test2)
R2.append(r2_test3)

ax.bar(X + 0.25, R2, color = 'r', width = 0.25, label = 'cvxopt')

ax.legend()


plt.show()

