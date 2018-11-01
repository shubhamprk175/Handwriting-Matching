import pandas as pd
import matplotlib.pyplot as plt
from data_splitting import *
import random


def _initialize_weights(length):
    return random.sample(range(200), length)

def _sigmoid(Z):
    return 1 / (1 + np.exp(-Z))

def _decision_boundary(prob):
    return np.asarray([ 1 if p >= .5 else 0 for p in prob])

def _calculate_accuracy(y_pred, y_test):
    correct, wrong = 0, 0
    for x, y in zip(y_pred, y_test):
        if x == y:
            correct += 1
        else:
            wrong += 1
    return correct/(correct+wrong)

def logistic_regression(X1, y1):

    # Returns dictionary containing training, testing, and val data and target
    dataset = split_data(X1.T, y1.T, algo='logistic')

    epoch = 100
    learningRate = 0.01
    W = _initialize_weights(np.size(X1, 1))


    for _ in range(epoch):
        Z = np.dot(W, np.transpose(dataset['X_train']))
        A = _sigmoid(np.asarray(Z))
        Delta = np.dot(np.transpose(dataset['X_train']), np.subtract(A, dataset['y_train']))
        W = np.add(W, np.dot(learningRate, Delta))


    y_pred = _sigmoid(np.dot(W, np.transpose(dataset['X_val'])))
    y_pred = _decision_boundary(y_pred).flatten()
    print("Validation Accuracy: {}".format(_calculate_accuracy(y_pred, dataset['y_val'])))

    y_pred = _sigmoid(np.dot(W, np.transpose(dataset['X_test'])))
    y_pred = _decision_boundary(y_pred).flatten()
    print("Testing Accuracy: {}".format(_calculate_accuracy(y_pred, dataset['y_test'])))
