import pandas as pd
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.callbacks import EarlyStopping, TensorBoard
from keras.optimizers import RMSprop, Adam

import numpy as np


def encodeLabel(labels):
    return np_utils.to_categorical(np.array(labels),2)



# ## Model Definition

input_size = X1.shape[1]
drop_out = 0.2
first_dense_layer_nodes  = 128
second_dense_layer_nodes = 64
third_dense_layer_nodes = 2

def get_model():

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



# # <font color='blue'>Creating Model</font>

# In[23]:


model = get_model()


# # <font color = blue>Run Model</font>

# In[25]:


validation_data_split = 0.2
num_epochs = 10000
model_batch_size = 128
tb_batch_size = 32
early_patience = 100

tensorboard_cb   = TensorBoard(log_dir='logs', batch_size= tb_batch_size, write_graph= True)
earlystopping_cb = EarlyStopping(monitor='val_loss', verbose=1, patience=early_patience, mode='min')

# Read Dataset
dataset, target = get_feature_matrix(data='hod', method='concatenate')

y_train = np.array(GenerateTrainingTarget(target, 80))
X_train   = np.transpose(GenerateTrainingDataMatrix(np.transpose(dataset), 80))
print("X_train.shape: {}".format(X_train.shape))
print("y_train.shape: {}".format(y_train.shape))

y_train = encodeLabel(y_train)


history = model.fit(X_train
                    , y_train
                    , validation_split=validation_data_split
                    , epochs=num_epochs
                    , batch_size=model_batch_size
                    , callbacks = [tensorboard_cb,earlystopping_cb]
                   )


# Training and Validation Graphs

# Testing Accuracy
def decodeLabel(encodedLabel):
    if encodedLabel == 0:
        return 0
    elif encodedLabel == 1:
        return 1


wrong   = 0
right   = 0

# testData = pd.read_csv('testing.csv')
y_test = np.array(GenerateValTargetVector(target, 20, (len(y_train))))
X_test = np.transpose(GenerateValData(np.transpose(dataset), 20, (len(y_train))))


processedTestData  = X_test
processedTestLabel = encodeLabel(y_test)
predictedTestLabel = []

for i,j in zip(processedTestData,processedTestLabel):
    y = model.predict(np.array(i).reshape(-1,processedTestData.shape[1]))
    predictedTestLabel.append(decodeLabel(y.argmax()))

    if j.argmax() == y.argmax():
        right = right + 1
    else:
        wrong = wrong + 1

print("Errors: " + str(wrong), " Correct :" + str(right))

print("Testing Accuracy: {0:.2f}%".format(right/(right+wrong)*100))
