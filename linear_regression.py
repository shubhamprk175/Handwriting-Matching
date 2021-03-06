import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from preprocessing import *
from data_splitting import *
from sklearn.cluster import KMeans
import math
import random


maxAcc = 0.0
maxIter = 0
C_Lambda = 0.03
TrainingPercent = 80
ValidationPercent = 10
TestPercent = 10
M = 5
PHI = []
IsSynthetic = False


def GenerateBigSigma(Data, MuMatrix,TrainingPercent,IsSynthetic):
    BigSigma    = np.zeros((len(Data),len(Data)))
    DataT       = np.transpose(Data)
    TrainingLen = math.ceil(len(DataT)*(TrainingPercent*0.01))
    varVect     = []
    for i in range(0,len(DataT[0])):
        vct = []
        for j in range(0,int(TrainingLen)):
            vct.append(Data[i][j])
        varVect.append(np.var(vct))

    for j in range(len(Data)):
        BigSigma[j][j] = varVect[j]
    if IsSynthetic == True:
        BigSigma = np.dot(3,BigSigma)
    else:
        BigSigma = np.dot(200,BigSigma)
    ##print ("BigSigma Generated..")
    return BigSigma

def GetScalar(DataRow, MuRow, BigSigInv):
    R = np.subtract(DataRow,MuRow)
    T = np.dot(BigSigInv,np.transpose(R))
    L = np.dot(R,T)
    return L

def GetRadialBasisOut(DataRow,MuRow, BigSigInv):
    phi_x = math.exp(-0.5*GetScalar(DataRow,MuRow,BigSigInv))
    return phi_x

def GetPhiMatrix(Data, MuMatrix, BigSigma, TrainingPercent = 80):
    DataT = np.transpose(Data)
    TrainingLen = math.ceil(len(DataT)*(TrainingPercent*0.01))
    PHI = np.zeros((int(TrainingLen),len(MuMatrix)))
    BigSigInv = np.linalg.inv(BigSigma)
    for  C in range(0,len(MuMatrix)):
        for R in range(0,int(TrainingLen)):
            PHI[R][C] = GetRadialBasisOut(DataT[R], MuMatrix[C], BigSigInv)
    #print ("PHI Generated..")
    return PHI

def GetWeightsClosedForm(PHI, T, Lambda):
    Lambda_I = np.identity(len(PHI[0]))
    for i in range(0,len(PHI[0])):
        Lambda_I[i][i] = Lambda
    PHI_T       = np.transpose(PHI)
    PHI_SQR     = np.dot(PHI_T,PHI)
    PHI_SQR_LI  = np.add(Lambda_I,PHI_SQR)
    PHI_SQR_INV = np.linalg.inv(PHI_SQR_LI)
    INTER       = np.dot(PHI_SQR_INV, PHI_T)
    W           = np.dot(INTER, T)
    ##print ("Training Weights Generated..")
    return W

def GetPhiMatrix(Data, MuMatrix, BigSigma, TrainingPercent = 80):
    DataT = np.transpose(Data)
    TrainingLen = math.ceil(len(DataT)*(TrainingPercent*0.01))
    PHI = np.zeros((int(TrainingLen),len(MuMatrix)))
    BigSigInv = np.linalg.inv(BigSigma)
    for  C in range(0,len(MuMatrix)):
        for R in range(0,int(TrainingLen)):
            PHI[R][C] = GetRadialBasisOut(DataT[R], MuMatrix[C], BigSigInv)
    #print ("PHI Generated..")
    return PHI

def GetValTest(VAL_PHI,W):
    Y = np.dot(W,np.transpose(VAL_PHI))
    ##print ("Test Out Generated..")
    return Y

def GetErms(VAL_TEST_OUT, y_val):
    sum = 0.0
    t=0
    accuracy = 0.0
    counter = 0
    val = 0.0
    for i in range (0,len(VAL_TEST_OUT)):
        sum = sum + math.pow((y_val[i] - VAL_TEST_OUT[i]),2)
        if(int(np.around(VAL_TEST_OUT[i], 0)) == y_val[i]):
            counter+=1
    accuracy = (float((counter*100))/float(len(VAL_TEST_OUT)))
    ##print ("Accuracy Generated..")
    ##print ("Validation E_RMS : " + str(math.sqrt(sum/len(VAL_TEST_OUT))))
    return (str(accuracy) + ',' +  str(math.sqrt(sum/len(VAL_TEST_OUT))))

def _plot_graph(x, y_train, y_val, y_test):
    print("Plotting")
    fig = plt.figure()
    plt.figure(1)
    plt.plot(x, y_train, 'r--', label="Training")
    plt.plot(x, y_val, 'g--', label="Validation")
    plt.plot(x, y_test, 'b--', label="Testing")
    plt.xlabel("LearningRate")
    plt.ylabel("E RMS")
    plt.legend(loc='upper left')
    plt.title("RMS")
    plt.show()

def linear_regression(X1, y1):

    RawTarget = y1.T
    RawData   = np.transpose(X1)

    dataset = split_data(RawData, RawTarget, algo='linear')
    dataset['X_train'].shape
    X1.shape
    dataset['X_val']


    ErmsArr = []
    AccuracyArr = []

    kmeans = KMeans(n_clusters=M, random_state=0).fit(np.transpose(dataset['X_train']))
    Mu = kmeans.cluster_centers_
    Mu.shape

    BigSigma     = GenerateBigSigma(RawData, Mu, TrainingPercent,IsSynthetic)
    TRAINING_PHI = GetPhiMatrix(RawData, Mu, BigSigma, TrainingPercent)
    W            = GetWeightsClosedForm(TRAINING_PHI,dataset['y_train'],(C_Lambda))
    TEST_PHI     = GetPhiMatrix(dataset['X_test'], Mu, BigSigma, 100)
    VAL_PHI      = GetPhiMatrix(dataset['X_val'], Mu, BigSigma, 100)




    print("Mu.shape: {}".format(Mu.shape))
    print("BigSigma.shape: {}".format(BigSigma.shape))
    print("TRAINING_PHI.shape: {}".format(TRAINING_PHI.shape))
    print("W.shape".format(W.shape))
    print("VAL_PHI.shape: {}".format(VAL_PHI.shape))
    print("TEST_PHI.shape: {}".format(TEST_PHI.shape))



    TR_TEST_OUT  = GetValTest(TRAINING_PHI,W)
    VAL_TEST_OUT = GetValTest(VAL_PHI,W)
    TEST_OUT     = GetValTest(TEST_PHI,W)




    print ('----------------------------------------------------')
    print ('--------------Please Wait for 2 mins!----------------')
    print ('----------------------------------------------------')




    W_Now        = np.dot(220, W)
    La           = 2
    learningRate = 0.01
    L_Erms_Val   = []
    L_Erms_TR    = []
    L_Erms_Test  = []
    W_Mat        = []

    test_acc = []
    val_acc = []
    train_acc = []
    LR_Arr = []
    for _ in range(5):
        for i in range(0,400):

            #print ('---------Iteration: ' + str(i) + '--------------')
            Delta_E_D     = -np.dot((dataset['y_train'][i] - np.dot(np.transpose(W_Now),TRAINING_PHI[i])),TRAINING_PHI[i])
            La_Delta_E_W  = np.dot(La,W_Now)
            Delta_E       = np.add(Delta_E_D,La_Delta_E_W)
            Delta_W       = -np.dot(learningRate,Delta_E)
            W_T_Next      = W_Now + Delta_W
            W_Now         = W_T_Next

            #-----------------TrainingData Accuracy---------------------#
            TR_TEST_OUT   = GetValTest(TRAINING_PHI,W_T_Next)
            Erms_TR       = GetErms(TR_TEST_OUT,dataset['y_train'])
            L_Erms_TR.append(float(Erms_TR.split(',')[1]))

            #-----------------ValidationData Accuracy---------------------#
            VAL_TEST_OUT  = GetValTest(VAL_PHI,W_T_Next)
            Erms_Val      = GetErms(VAL_TEST_OUT,dataset['y_val'])
            L_Erms_Val.append(float(Erms_Val.split(',')[1]))

            #-----------------TestingData Accuracy---------------------#
            TEST_OUT      = GetValTest(TEST_PHI,W_T_Next)
            Erms_Test = GetErms(TEST_OUT,dataset['y_test'])
            L_Erms_Test.append(float(Erms_Test.split(',')[1]))




        print ('----------Gradient Descent Solution--------------------')
        print ("M = 10 \nLambda  = 0.0001\neta=0.01")
        print ("E_rms Training   = " + str(np.around(min(L_Erms_TR),5)))
        print ("E_rms Validation = " + str(np.around(min(L_Erms_Val),5)))
        print ("E_rms Testing    = " + str(np.around(min(L_Erms_Test),5)))

        learningRate *= 2
        test_acc.append(np.around(min(L_Erms_TR)))
        val_acc.append(np.around(min(L_Erms_Val)))
        train_acc.append(np.around(min(L_Erms_Test)))
        LR_Arr.append(learningRate)

    _plot_graph(LR_Arr, test_acc, val_acc, train_acc)
