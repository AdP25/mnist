import numpy as np
import pandas as pd

data = pd.read_csv('train.csv')
data.head()

data = np.array(data)
np.random.shuffle(data)
r,c = data.shape
data_train = data[0:1000].T
X_train = data_train[1:c]
Y_train = data_train[0]

data_val = data[1000:r].T
X_val = data_val[1:c]
Y_val = data_val[0]

# 0th col of all rows
X_val[:,0].shape

def init_params(size):
    W1 = np.random.rand(10, size) - 0.5
    b1 = np.random.rand(10,1) - 0.5

    W2 = np.random.rand(10, 10) - 0.5
    b2 = np.random.rand(10, 1) - 0.5
    return W1, b1, W2, b2

def reLu(Z):
    return np.maximum(0, Z)

def derivative_ReLU(Z):
    return Z > 0

def softmax(Z):
    # Subtracting the maximum value helps prevent potential issues like numeric overflow when 
    # exponentiating large numbers, maintaining numerical stability.
    exp = np.exp(Z - np.max(Z))
    return exp / exp.sum(axis=0)

# A one-hot array is a representation used in machine learning, particularly in classification tasks, 
# to encode categorical variables into a binary vector format. It's a way to represent categorical data where 
# each category is converted into a vector of binary values, with only one element being 'hot' (having the value 1)
# while the rest are 'cold' (having the value 0).
def one_hot_arr(Y):
    one_hot_Y = np.zeros((Y.size, Y.max() + 1))
    print(one_hot_Y.shape)
    ind_arr = np.arange(Y.size)
    print(ind_arr)
    # row from ind_arr and col from Y arr will be set to 1
    one_hot_Y[ind_arr, Y] = 1
    print(one_hot_Y)
    one_hot_Y = one_hot_Y.T
    return one_hot_Y

def forw_prop(X, W1, b1, W2, b2):
    Z1 = W1.dot(X) + b1 # 10, m
    print("Z1 : ", Z1)
    A1 = reLu(Z1) # 10,m
    Z2 = W2.dot(A1) + b2 # 10,m
    A2 = softmax(Z2) # 10,m
    return Z1, A1, Z2, A2

def back_prop(X, Y, A1, A2, W2, Z1, m):
    one_hot_Y = one_hot_arr(Y)
    dZ2 = 2 * (A2 - one_hot_Y) # 10, m
    dW2 = 1/m * (dZ2.dot(A1.T)) # 10 , 10
    db2 = 1/m * np.sum(dZ2,1) # 10, 1
    dZ1 = W2.T.dot(dZ2) * derivative_ReLU(Z1) # 10, m
    dW1 = 1/m * (dZ1.dot(X.T)) # 10, 784
    db1 = 1/m * np.sum(dZ1,1) # 10, 1

    return dW1, db1, dW2, db2

def update_params(alpha, W1, b1, W2, b2, dW1, db1, dW2, db2):
    W1 -= alpha * dW1
    b1 -= alpha * np.reshape(db1, (10,1))
    W2 -= alpha * dW2
    b2 -= alpha * np.reshape(db2, (10,1))

    return W1, b1, W2, b2

def get_predictions(A2):
    # returns the indices of the maximum values along a specified axis
    return np.argmax(A2, 0)

def get_accuracy(predictions, Y):
    return np.sum(predictions == Y)/Y.size

# get your weights and biases learnt
def gradient_descent(X, Y, alpha, iterations):
    # size - number of features ,
    # m - number of samples from the input data X
    size , m = X.shape

    W1, b1, W2, b2 = init_params(size)
    for i in range(iterations):
        Z1, A1, Z2, A2 = forw_prop(X, W1, b1, W2, b2)
        print("A2 : ", A2)
        dW1, db1, dW2, db2 = back_prop(X, Y, A1, A2, W2, Z1, m)

        W1, b1, W2, b2 = update_params(alpha, W1, b1, W2, b2, dW1, db1, dW2, db2)   
        print("W1 : ", W1)
        if (i+1) % int(iterations/10) == 0:
            print(f"Iteration: {i+1} / {iterations}")
            prediction = get_predictions(A2)
            print(f'{get_accuracy(prediction, Y):.3%}')
    return W1, b1, W2, b2

def make_predictions(X, W1, b1, W2, b2):
    _, _, _, A2 = forw_prop(X, W1, b1, W2, b2)
    predictions = get_predictions(A2)
    return predictions

def show_prediction(index, X, Y, W1, b1, W2, b2):
    vect_X = X[:, index, None]
    prediction = make_predictions(vect_X, W1, b1, W2, b2)
    label = Y[index]
    print("Prediction: ", prediction)
    print("Label: ", label)
     
    # current_image = current_image.reshape((28, 28)) * 255
    # plt.gray()
    # plt.imshow(current_image, interpolation='nearest')
    # plt.show()

# --------- MAIN ---------

# before training
W1, b1, W2, b2 = init_params(784)
show_prediction(0, X_val, Y_val, W1, b1, W2, b2)
# show_prediction(1, X_val, Y_val, W1, b1, W2, b2)
# show_prediction(2, X_val, Y_val, W1, b1, W2, b2)
# show_prediction(3, X_val, Y_val, W1, b1, W2, b2)

# after training
W1, b1, W2, b2 = gradient_descent(X_train, Y_train, 0.10, 10)
print("X_train : ", X_train.shape)
#print("Y_train : ", Y_train)
print("W1 : ", W1.shape)
print("b1 : ", b1.shape)
print("W2 : ", W2.shape)
print("b2 : ", b2.shape)

show_prediction(0, X_val, Y_val, W1, b1, W2, b2)
# show_prediction(1, X_val, Y_val, W1, b1, W2, b2)
# show_prediction(2, X_val, Y_val, W1, b1, W2, b2)
# show_prediction(3, X_val, Y_val, W1, b1, W2, b2)




# Y = np.array([2,0,1,1,2,0])
# print(one_hot_arr(Y))
# dev_predictions = make_predictions(X_dev, W1, b1, W2, b2)
# get_accuracy(dev_predictions, Y_dev)

