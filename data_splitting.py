import numpy as np
import math



def _training_target(rawTraining,TrainingPercent = 80):
    TrainingLen = int(math.ceil(len(rawTraining)*(TrainingPercent*0.01)))
    t           = rawTraining[:TrainingLen]
    #print(str(TrainingPercent) + "% Training Target Generated..")
    return t

def _training_data(rawData, TrainingPercent = 80):
    T_len = int(math.ceil(len(rawData[0])*0.01*TrainingPercent))
    d2 = rawData[:,0:T_len]
    #print(str(TrainingPercent) + "% Training Data Generated..")
    return d2

def _val_data(rawData, ValPercent, TrainingCount):
    valSize = int(math.ceil(len(rawData[0])*ValPercent*0.01))
    V_End = TrainingCount + valSize
    dataMatrix = rawData[:, TrainingCount+1:V_End]
    #print (str(ValPercent) + "% Val Data Generated..")
    return dataMatrix

def _val_target(rawData, ValPercent, TrainingCount):
    valSize = int(math.ceil(len(rawData)*ValPercent*0.01))
    V_End = TrainingCount + valSize
    t =rawData[TrainingCount+1:V_End]
    #print (str(ValPercent) + "% Val Target Data Generated..")
    return t


def split_data(X1, y1, algo='linear'):
    y_train = np.array(_training_target(y1, 80))
    # removed np.transpose(X1)
    X_train   = _training_data(X1, 80)
    print("X_train.shape: {}".format(X_train.shape))
    print("y_train.shape: {}".format(y_train.shape))

    y_val = np.array(_val_target(y1,10, (len(y_train))))
    X_val = _val_data(X1, 10, (len(y_train)))
    print("X_val.shape: {}".format(X_val.shape))
    print("y_val.shape: {}".format(y_val.shape))


    y_test = np.array( _val_target(y1, 10, (len(y_train)+len(X_val))))
    X_test = _val_data(X1, 10, (len(y_train)+len(X_val)))
    print("X_test.shape: {}".format(X_test.shape))
    print("y_test.shape: {}".format(y_test.shape))

    if algo == 'linear':
        return {'X_train':X_train, 'y_train':y_train, 'X_val':X_val, 'y_val':y_val, 'X_test':X_test, 'y_test':y_test}
    else:
        return {'X_train':X_train.T, 'y_train':y_train.T, 'X_val':X_val.T, 'y_val':y_val.T, 'X_test':X_test.T, 'y_test':y_test.T}
