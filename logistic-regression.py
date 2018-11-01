import pandas as pd
import matplotlib.pyplot as plt
from preprocessing import *
from data_splitting import *
import random


def initialize_weights(length):
    return random.sample(range(200), length)

def sigmoid(Z):
  return 1 / (1 + np.exp(-Z))

X1, y1 = get_feature_matrix(data='hod', method='concatenate')
# Returns dictionary containing training, testing, and val data and target
dataset = split_data(X1.T, y1)

epoch = 100
learningRate = 0.01
W = initialize_weights(np.size(X1, 1))


for _ in range(epoch):
    Z = np.dot(W, np.transpose(dataset['X_train']))
    A = sigmoid(np.asarray(Z))
    Delta = np.dot(np.transpose(dataset['X_train']), np.subtract(A, dataset['y_train']))
    W = np.add(W, np.dot(learningRate, Delta))


y_pred = sigmoid(np.dot(W, np.transpose(dataset['X_test'])))
print("y_pred[0:10]: {}".format(y_pred[0:10]))
print("y_test[0:10]: {}".format(dataset['y_test'][0:10]))
