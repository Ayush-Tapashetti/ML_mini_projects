import numpy as np
import pandas as pd

#1.0 Data Loading
data = pd.read_csv('TRAIN.csv')
data = np.array(data)
m, n = data.shape
np.random.shuffle(data)

#1.1 Cross Validation
data_dev = data[0:1000].T
Y_dev = data_dev[0]
X_dev = data_dev[1:n]
X_dev = X_dev / 255.

#1.2 Training Set
data_train = data[1000:m].T
Y_train = data_train[0]
X_train = data_train[1:n]
X_train = X_train / 255.
_, m_train = X_train.shape

#1.3 Human Readable Label Names
LABELS = {
    0: "T-shirt", 1: "Trouser",
    2: "Pullover", 3: "Dress",
    4: "Coat", 5: "Sandal",
    6: "Shirt", 7: "Sneaker",
    8: "Bag", 9: "ANkleBoot",
}

#2.0 Initalise Parameters
def init_params():
    W1 = np.random.rand(128, 784) - 0.5
    b1 = np.random.rand(128, 1) - 0.5
    W2 = np.random.rand(10, 128) - 0.5
    b2 = np.random.rand(10, 1) - 0.5
    return W1, b1, W2, b2

#3.0 Activation Functions
def ReLu(Z):
    return np.maximum(Z, 0)

def softmax(Z):
    A = np.exp(Z) / np.sum(np.exp(Z), axis=0)
    return A

def  deriv_ReLu(Z):
    return Z>0        

#4.0 Forward Propagation
def Forward_Prop(W1, b1, W2, b2, X):
    Z1 = W1.dot(X)+b1
    A1 = ReLu(Z1)
    Z2 = W2.dot(A1)+b2
    A2 = softmax(Z2)
    return Z1, A1, Z2, A2

#5.0 Back Propagation
def one_hot(Y):
    one_hot_Y = np.zeros((Y.size, Y.max()+1))
    one_hot_Y[np.arange(Y.size), Y] = 1
    one_hot_Y = one_hot_Y.T
    return one_hot_Y

def back_prop(Z1, A1, Z2, A2, W1, W2, X, Y):
    m = Y.size
    one_hot_Y = one_hot(Y)
    dZ2 = A2 - one_hot_Y
    dW2 = (1 / m) * dZ2.dot(A1.T)
    db2 = (1 / m) *np.sum(dZ2, axis=1,keepdims=True)
    dZ1 = W2.T.dot(dZ2) * deriv_ReLu(Z1)
    dW1 = (1 / m) * dZ1.dot(X.T)
    db1 = (1 / m) * np.sum(dZ1, axis=1, keepdims=True)
    return dW1,db1, dW2, db2

#6.0 Update Parameters
def update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha):
    W1 = W1 - (alpha * dW1)
    b1 = b1 - (alpha * db1)
    W2 = W2 - (alpha * dW2)
    b2 = b2 - (alpha * db2)
    return W1, b1, W2, b2

#7.0 utils & training
def get_predi(A2):
    return np.argmax(A2, 0)

def get_accu(predictions, Y):
    return np.sum(predictions == Y) / Y.size

def gradient_Descent(X, Y, alpha, iterations):
    W1, b1, W2, b2 = init_params()
    for i in  range(iterations):
        Z1, A1, Z2, A2 = Forward_Prop(W1, b1, W2, b2, X)
        dW1, db1, dW2, db2 = back_prop(Z1, A1, Z2, A2, W1, W2, X, Y)
        W1, b1, W2, b2 = update_params(W1, b1, W2, b2, dW1, db1, dW2,db2, alpha)
        if i%10==0:
            print("Iterations", i)
            predictions = get_predi(A2)
            print("Accuracy:", get_accu(predictions, Y))
    return W1, b1, W2, b2    

W1, b1, W2, b2 = gradient_Descent(X_train, Y_train, 0.10, 1000)

#8.0 Evalute on set
def make_predictions(X, W1, b1, W2, b2):
    _, _, _, A2 = Forward_Prop(W1, b1, W2, b2, X)
    predictions = get_predi(A2)
    return predictions

dev_predictions = make_predictions(X_dev, W1, b1, W2, b2)
print("Dev set accuracy:", get_accu(dev_predictions, Y_dev))

#9.0 Inspect prediction
import matplotlib.pyplot as plt

def test_prediction(index, W1, b1, W2, b2):
    current_image = X_train[:,  index, None]
    prediction = make_predictions(X_train[:, index, None], W1, b1, W2, b2)
    label = Y_train[index]
    print("Prediction:", LABELS[prediction[0]])
    print("Actual label:", LABELS[label])

    current_image = current_image.reshape((28,28)) * 255
    plt.gray()
    plt.imshow(current_image, interpolation = 'nearest')
    plt.show()

test_prediction(0, W1, b1, W2, b2)
