import pandas as pd
import numpy as np
from data_splitting import *

from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.callbacks import EarlyStopping, TensorBoard
from keras.optimizers import RMSprop, Adam




def _encode_label(labels):
    return np_utils.to_categorical(np.array(labels),2)


# ## Model Definition
def _get_model(input_size):

    drop_out = 0.2
    first_dense_layer_nodes  = 128
    second_dense_layer_nodes = 64
    third_dense_layer_nodes = 2

    model = Sequential()

    model.add(Dense(first_dense_layer_nodes, input_dim=input_size))
    model.add(Activation('sigmoid'))
    model.add(Dropout(drop_out))

    model.add(Dense(second_dense_layer_nodes))
    model.add(Activation('sigmoid'))
    model.add(Dropout(drop_out))

    model.add(Dense(third_dense_layer_nodes))
    model.add(Activation('softmax'))

    model.summary()

    opt = Adam()
    model.compile(optimizer=opt,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model


def neural_network(X1, y1):
    model = _get_model(X1.shape[1])

    validation_data_split = 0.2
    num_epochs = 10000
    model_batch_size = 256
    tb_batch_size = 128
    early_patience = 100

    tensorboard_cb   = TensorBoard(log_dir='logs', batch_size= tb_batch_size, write_graph= True)
    earlystopping_cb = EarlyStopping(monitor='val_loss', verbose=1, patience=early_patience, mode='min')

    dataset = split_data(X1.T, y1.T, algo='neural')
    print("dataset['X_train'].shape: {}".format(dataset['X_train'].shape))
    print("dataset['y_train'].shape: {}".format(dataset['y_train'].shape))

    dataset['y_train'] = _encode_label(dataset['y_train'])


    history = model.fit(dataset['X_train']
                        , dataset['y_train']
                        , validation_split=validation_data_split
                        , epochs=num_epochs
                        , batch_size=model_batch_size
                        , callbacks = [tensorboard_cb,earlystopping_cb]
                       )

    _testing_accuracy(dataset['X_test'], dataset['y_test'], model)

    # Training and Validation Graphs

def _decode_label(encodedLabel):
    if encodedLabel == 0:
        return 0
    elif encodedLabel == 1:
        return 1


# Testing Accuracy
def _testing_accuracy(X_test, y_test, model):
    wrong   = 0
    right   = 0


    processedTestData  = X_test
    processedTestLabel = _encode_label(y_test)
    predictedTestLabel = []

    for i,j in zip(processedTestData,processedTestLabel):
        y = model.predict(np.array(i).reshape(-1,processedTestData.shape[1]))
        predictedTestLabel.append(_decode_label(y.argmax()))

        if j.argmax() == y.argmax():
            right = right + 1
        else:
            wrong = wrong + 1

    print("Errors: " + str(wrong), " Correct :" + str(right))

    print("Testing Accuracy: {0:.2f}%".format(right/(right+wrong)*100))
