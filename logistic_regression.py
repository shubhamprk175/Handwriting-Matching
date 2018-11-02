import pandas as pd
import matplotlib.pyplot as plt
from data_splitting import *
import random
import math

def _initialize_weights(length):
    return random.sample(range(length), length)

def _sigmoid(Z):
    A = np.divide(1.0, np.add(1, np.exp(-Z)))
    return A

def _decision_boundary(prob):
    return np.asarray([ 1 if p >= .5 else 0 for p in prob])

def _calculate_accuracy(y_pred, y_test):
    correct, wrong = 1, 1
    for x, y in zip(y_pred, y_test):
        if x == y:
            correct += 1
        else:
            wrong += 1
    return correct/(correct+wrong)

def _plot_graph(x, y_test, y_val, y_label1, y_label2):
    print("Plotting")
    fig = plt.figure()
    plt.figure(1)
    plt.plot(x, y_test, 'r-', label=y_label1)
    plt.plot(x, y_val, 'g-', label=y_label2)
    plt.xlabel("Learning Rate")
    plt.ylabel("Accuracy")
    plt.legend(loc='upper left')
    plt.title("Logistic Regression")
    plt.show()

def logistic_regression(X1, y1):

    # Returns dictionary containing training, testing, and val data and target
    dataset = split_data(X1.T, y1.T, algo='logistic')

    epoch = 1000
    Lambda = 2
    print("Epoch: {}".format(epoch))
    learningRate = 0.05
    W = _initialize_weights(np.size(X1, 1))
    test_acc = []
    val_acc = []
    Lambda_Arr = []
    for i in range(5):

        for _ in range(epoch):
            Z = np.dot(W, np.transpose(dataset['X_train']))
            A = _sigmoid(np.asarray(Z))
            Delta = np.multiply(np.add(np.dot(np.transpose(dataset['X_train']), np.subtract(A, dataset['y_train'])), np.multiply(Lambda,W)),1./np.size(X1))
            W = np.subtract(W, np.dot(learningRate, Delta))


        y_pred = _sigmoid(np.dot(W, np.transpose(dataset['X_val'])))
        y_pred = _decision_boundary(y_pred).flatten()
        val_acc.append(_calculate_accuracy(y_pred, dataset['y_val']))
        Lambda_Arr.append(learningRate)
        print("Testing Accuracy: {0}, Lambda: {1}".format(val_acc[i], learningRate))


        y_pred = _sigmoid(np.dot(W, np.transpose(dataset['X_test'])))
        y_pred = _decision_boundary(y_pred).flatten()
        test_acc.append(_calculate_accuracy(y_pred, dataset['y_test']))
        print("Testing Accuracy: {0}, Lambda: {1}".format(test_acc[i], learningRate))

        learningRate *= 2

    _plot_graph(Lambda_Arr, test_acc, val_acc, y_label1="Testing", y_label2="Validation")
