
import numpy as np # linear algebra
import pandas as pd
import matplotlib.pyplot as mp
import random
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

def activation_fn(x):
#     if x > 0:
#         return x
#     else:
#         return 0
    return 1- 2/(np.exp(2*x) + 1)
    
def deriv_activation_fn(x):
    return (1-np.square(x))
#     if x > 0:
#         return 1
#     else:
#         return 0


class layer:
    def __init__(self, neuron_number, prev_layer_number):
        self.neurons_wt = np.random.uniform(low = -0.1, high = 0.1, size = (neuron_number, prev_layer_number+1))
        self.neuron_wt_changes = np.zeros((neuron_number, prev_layer_number+1))
        
    def set_layer_activation1(self,input):   
        self.layer_activation = np.vectorize(activation_fn)(np.dot(self.neurons_wt,np.append(input, [1])))

        
    def set_layer_activation(self,inputl):
        #print(inputl.shape)
        inputl = np.append(inputl,np.ones([inputl.shape[0],1]),axis=1)
        self.layer_activation = np.dot(inputl,self.neurons_wt.T)
        self.layer_activation = activation_fn(self.layer_activation)
#         print(self.layer_activation)
        return self.layer_activation
#         self.layer_activation = np.vectorize(activation_fn)(np.dot(self.neurons_wt,np.append(inputl, [1])))
        #this is a column matrix of height equal to the number of neurons. Assuming that input is a row matrix.
        
    def set_layer_error(self, neuron_error):
        self.layer_error = neuron_error #this is a row vector with no. of columns= no of neurons
        
    def get_layer_activation(self):
        return self.layer_activation
    
    def print_layer(self):
        print(self.layer_activation)


# In[189]:


class network:
    def __init__(self, number_of_layers, number_of_neurons): #number of layers includes the hidden layers and output layer but
        #number_of_neurons includes the number of inputs and the subsequent number of neurons in diff layers
        self.structure = np.array([layer(number_of_neurons[i], number_of_neurons[i-1]) for i in range (1, number_of_layers+1)])
    
    def fpropagate1 (self, inputl):    
        x = np.shape(self.structure)[0]
        in1 = inputl
        for i in range(x):
            self.structure[i].set_layer_activation1(in1)
            in1 = self.structure[i].get_layer_activation()
    
    def fpropagate (self, inputl):    
        x = np.shape(self.structure)[0]
        in1 = inputl
        for i in range(x):
            in1 =self.structure[i].set_layer_activation(in1)
            #print(np.shape(in1))
#             in1 = self.structure[i].get_layer_activation()
            
    def print_output(self):
        self.structure[-1].print_layer()
        
    def output_error_calc(self, output):
        layer_last = self.structure[-1]
        a = layer_last.get_layer_activation() #row vector or matrix of shape batchsize x 10
#         b = np.zeros(a.shape)
        b = np.eye(a.shape[1])[output]
        layer_last.error = (np.square(b-a)).mean(axis=1).sum()
#         print(layer_last.error)
        layer_last.set_layer_error(np.multiply((b-a),deriv_activation_fn(a)))
        
    def hidden_error_calc(self):
        for i in reversed(range(np.shape(self.structure)[0]-1)):
            layeri = self.structure[i]
            layeri1 = self.structure[i+1]
            deltai = np.dot(layeri1.layer_error,layeri1.neurons_wt)
            deltai = deltai[:,:-1]
#             print("hid_err_calc",layeri1.layer_error.shape,layeri1.neurons_wt.shape)
            layeri.set_layer_error(np.multiply(deltai, np.vectorize(deriv_activation_fn)(layeri.get_layer_activation())))
                            
    def input_weights_changes(self, inputl, lr):
        first_layer = self.structure[0]
        inputl = np.append(inputl,np.ones([inputl.shape[0],1]),axis=1)
        first_layer.neuron_wt_changes = lr*np.dot( first_layer.layer_error.T,inputl)
    
    def hidden_weights_change(self, lr):
        for i in range(1, np.shape(self.structure)[0]):
            layeri = self.structure[i]
            layeri1 = self.structure[i-1]
            inputl = np.append(layeri1.get_layer_activation(),np.ones([layeri1.get_layer_activation().shape[0],1]),axis=1)
#             layeri.neuron_wt_changes = layeri.neuron_wt_changes + lr*layeri.layer_error.T*np.append(layeri1.get_layer_activation(), [1])
            layeri.neuron_wt_changes = lr*np.dot(layeri.layer_error.T,inputl) 
#             print(layeri.neuron_wt_changes)
                
    def error(self, output):
        layer_last = self.structure[-1]
        a = layer_last.get_layer_activation()
        b = np.zeros((1, np.size(a)))
        b[0, output] = 1
        return np.sum(np.square(b-a))
    
    def argMax(self, y):
        #print(y)
        #print(np.argmax(self.structure[-1].get_layer_activation()))
        if y == np.argmax(self.structure[-1].get_layer_activation()):
            return True
        else:
            return False
    def getAccuracy(self,y):
        layer_last = self.structure[-1]
        a = layer_last.get_layer_activation()
        label = np.argmax(a,axis=1)
        correct = np.equal(label,y).sum()
        acc = (correct/np.size(y))*100
        return acc
    
    def output_(self):
        layer_last = self.structure[-1]
        a = layer_last.get_layer_activation()
        label = np.argmax(a)
        return label


def back_propagation(n, inputl, output, lr): #over a batch
    x = np.shape(inputl)[0]
    
#     for i in range(x):
    n.fpropagate(inputl) 
    n.output_error_calc(output) #error calculation for each neuron #error for output layer calculated separately
    n.hidden_error_calc() #calculation of error for hidden layers
    n.input_weights_changes(inputl, lr) #changing the input weights separately
    n.hidden_weights_change(lr) #changing the weights connected to second hidden layer onwards till output layer
    for j in range(np.shape(n.structure)[0]):
        n.structure[j].neurons_wt = n.structure[j].neurons_wt + n.structure[j].neuron_wt_changes/(x*np.shape(n.structure[j].neurons_wt)[0])
    # all weights changed!


data_train = pd.read_csv('/kaggle/input/ell409-assignment-1/mnist_train.csv')
# from random import shuffle
d_train = data_train.iloc[0:5000, :]
x_train = np.array(data_train.iloc[0:5000, 1:])/255
y_train = np.array(data_train.iloc[0:5000, 0])
# ind_list = [i for i in range(x_train.shape[0])]
# shuffle(ind_list)
# x_train = x_train[ind_list, :]
# y_train = y_train[ind_list,]

# # shuffleInd = random.sample(range(0, x_train.shape[0]), x_train.shape[0])
# # x_train  =  np.array([x_train[i,:] for i in shuffleInd])
# # y_train  =  np.array([y_train[i,:] for i in shuffleInd])
x_validate = data_train.iloc[5000:7000, 1:]/255
y_validate = data_train.iloc[5000:7000, 0]

Network = network(2, [784, 128, 10])
epoch = 100
batch_size = 100
learning_rate = 2
#training_error = np.zeros((epoch, 1))
#validation_error = np.zeros((epoch, 1))
accuracy_t = np.zeros(epoch)
accuracy_v = np.zeros(epoch)

for b in range(epoch):
    i = 0
    d_train = d_train.sample(frac = 1).reset_index(drop = True)
    x_train = np.array(d_train.iloc[:,1:])/255
    y_train = np.array(d_train.iloc[:, 0])
    while i < np.shape(x_train)[0]:
        #print(i)
        if (i + batch_size <= np.shape(x_train)[0]):
            back_propagation(Network, x_train[i:i+batch_size], y_train[i:i+batch_size], learning_rate) #doubtful of range. Check out!
        else:
            back_propagation(Network, x_train[i:np.shape(x_train)[0]], y_train[i:np.shape(x_train)[0]], learning_rate)
        i = i + batch_size
    print("Epoch:",b,end=' ')
    Network.fpropagate(x_train)
    n = Network.getAccuracy(y_train)
    accuracy_t[b] = n
    print(n,end=' ')
    Network.fpropagate(x_validate)
    m = Network.getAccuracy(y_validate)
    accuracy_v[b] = m 
    print(m, end = '\n')
    
x = np.arange(0, epoch)  # for plotting wrt epochs
fig = mp.figure()
ax = mp.subplot(111)
ax.plot(x, accuracy_t, label = 'training accuracy')
ax.plot(x, accuracy_v, label = 'validation accuracy')
mp.title('Number of epochs vs Accuracy plot')
mp.xlabel('number of epochs')
mp.ylabel('accuracy')
fig.savefig('test.jpg')
ax.legend()
mp.show()
'''#     print('backprop done')
    print("Epoch:",b,end=' ')
    Network.fpropagate(x_train)
    print(Network.getAccuracy(y_train),end='\n')'''
    
d_test = pd.read_csv('../input/mnist-data-set/mnist_test.csv')
print(np.shape(d_test))
x_test = np.array(d_test.iloc[:,:])/255
print(np.shape(x_test))
predictions = []
for s in range(np.shape(x_test)[0]):
    Network.fpropagate1(x_test[s, :])
                       ######
    predictions.append(Network.output_())
submission = pd.DataFrame({"id": list(range(1,len(predictions)+1)),
                         "label": predictions})
submission.to_csv("DR1.csv", index=False, header=True)

Network.fpropagate(x_validate)
print(Network.getAccuracy(y_validate))

'''for x in range(20):
    for b in range(epoch):
        i = 0
        d_train = d_train.sample(frac = 1).reset_index(drop = True)
        x_train = np.array(d_train.iloc[:,1:])/255
        y_train = np.array(d_train.iloc[:, 0])
        while i < np.shape(x_train)[0]:
            #print(i)
            if (i + batch_size <= np.shape(x_train)[0]):
                back_propagation(Network, x_train[i:i+batch_size], y_train[i:i+batch_size], u[x]) #doubtful of range. Check out!
            else:
                back_propagation(Network, x_train[i:np.shape(x_train)[0]], y_train[i:np.shape(x_train)[0]], u[x])
            i = i + batch_size
       # t1 = 0
       # v = 0
    Network.fpropagate(x_train)
    n = Network.getAccuracy(y_train)
    print("Learning rate:",float(x/20),end=' ')
    print(n,end=' ')
    accuracy_lr_t[x] = n
    Network.fpropagate(x_validate)
    m = Network.getAccuracy(y_validate)    
    accuracy_lr_v[x] = m
    print(m,end='\n ')
   
    
#     print('backprop done')
    print("Epoch:",b,end=' ')
    Network.fpropagate(x_train)
    n = Network.getAccuracy(y_train)
    accuracy_t[b] = n
    print(n,end=' ')
    #print(np.shape(x_validate))
    Network.fpropagate(x_validate)
    m = Network.getAccuracy(y_validate)
    accuracy_v[b] = m 
    print(m, end = '\n')
    


x = np.arange(0, epoch, 1)  # for plotting wrt epochs
fig = mp.figure()
ax = mp.subplot(111)
ax.plot(x, accuracy_t, label = 'training accuracy')
ax.plot(x, accuracy_v, label = 'validation accuracy')
mp.title('Number of epochs vs Accuracy plot')
mp.xlabel('number of epochs')
mp.ylabel('accuracy')
fig.savefig('test.jpg')
ax.legend()
mp.show()

x = np.arange(0, 1, 0.05)
fig = mp.figure()
ax = mp.subplot(111)
ax.plot(x, accuracy_lr_t, label = 'training accuracy')
ax.plot(x, accuracy_lr_v, label = 'validation accuracy')
mp.title('Learning rate vs accuracy plot')
mp.xlabel('Learning rate')
mp.ylabel('accuracy')
fig.savefig('test.jpg')
ax.legend()
mp.show()
'''
