import numpy as np
import pandas as pd

#1.0 Data Loading

data = pd.read_csv('train.csv')
data = np.array(data)
m, n = data.shape
np.random.shuffle(data)

#1.1 CrossValidation
data_1= data[0:1000].T
Y_1 = data_1[0]
X_1 = data_1[1:n]
X_1 = X_1 / 255

#1.2 Training
data_train = data[1000:m].T
Y_train = data_train[0]
X_train = data_train[1:n]
X_train = X_train / 255
m_train = X_train.shape

#2.0 Initialization of Parameters
def init_par():
    W1 = np.random.rand(10, 784) - 0.5
    b1 = np.random.rand(10, 1) - 0.5
    W2 = np.random.rand(10, 10) - 0.5
    b2 = np.random.rand(10, 1) - 0.5 
    return W1, b1, W2, b2

#3.0 Activation Functions
def Relu(Z):
    return np.maximum(Z,0)

def tomax(Z):
    A = np.exp(Z)/sum(np.exp(Z))
    return A

def deriv_Relu(Z):
    return Z>0

#4.0 Forward Propagaiton
def forward_prop(W1, b1, W2, b2, X):
    Z1 = W1.dot(X)+b1
    A1 = Relu(Z1)

    Z2 = W2.dot(A1)+b2
    A2 = tomax(Z2)
    return Z1, A1, Z2, A2

#5.0 Back Propagation
def one(Y):
    one_Y = np.zeros((Y.size, Y.max()+1))
    one_Y[np.arange(Y.size), Y] = 1
    one_Y = one_Y.T
    return one_Y

def back_prop(Z1, A1, Z2, A2, W1, W2, X, Y):
    one_Y = one(Y)
    m_batch = Y.size
    dZ2 = A2- one_Y
    dW2 = (1 / m_batch) * dZ2.dot(A1.T)
    db2 = (1 / m_batch) * np.sum(dZ2, axis=1, keepdims=True)
    dZ1 = W2.T.dot(dZ2) * deriv_Relu(Z1)
    dW1 = (1 / m_batch) * dZ1.dot(X.T)
    db1 = (1 / m_batch) * np.sum(dZ1, axis=1, keepdims=True)
    return dW1, db1, dW2, db2

#6 Parameters Updating...
def update_para(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha):
    W1 = W1 - alpha * dW1
    b1 = b1 - alpha * db1
    W2 = W2 - alpha * dW2
    b2 = b2 - alpha * db2
    return W1, b1, W2, b2

#7 Training Looop
def get_pred(A2):
    return np.argmax(A2, 0)

def get_acc(pred, Y):
    return np.sum(pred == Y)/(Y.size)

def gradient_descent(X, Y, alpha, ite):
    W1, b1, W2, b2 = init_par()
    for i in range(ite):
        Z1, A1, Z2, A2 = forward_prop(W1, b1, W2, b2, X)
        dW1, db1, dW2, db2 = back_prop(Z1, A1, Z2, A2, W1, W2, X, Y)
        W1, b1, W2, b2 = update_para(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha)
        if  i % 10 == 0:
            print("Iteration:", i)
            pred = get_pred(A2)
            print("Accuracy", get_acc(pred, Y))
    return W1, b1, W2, b2

#RUN TRAINING
W1, b1, W2, b2 = gradient_descent(X_train, Y_train, 0.10, 500)
