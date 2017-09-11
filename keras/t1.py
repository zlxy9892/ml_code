# -*- coding: utf-8 -*-

# test the library 'keras'

import numpy as np
import matplotlib.pyplot as plt
import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout


def loadDataSet():
    dataMat = []
    labelMat = []
    fr = open('./data/testSet.txt', 'r')
    lines = fr.readlines()
    for line in lines:
        strArr = line.strip().split()
        dataMat.append([float(strArr[0]), float(strArr[1])])
        label = float(strArr[2])
        if label == 0.0:
            labelMat.append([1.0, 0.0])
        else:
            labelMat.append([0.0, 1.0])
    #dataMat = np.mat(dataMat)
    #labelMat = np.mat(labelMat).reshape(np.shape(dataMat)[0], 1)
    return np.array(dataMat), np.array(labelMat)

def showData(X, y):
    plt.scatter(X[:,0], X[:,1], c=y, edgecolor='k')
    plt.show()


    
# ----------------------- main --------------------------- #
X, y = loadDataSet()
#showData(X, y)

model = Sequential()

inputDim = 2
outputDim = 2

model.add(Dense(4, activation='relu', input_dim=inputDim))
model.add(Dense(outputDim, activation='sigmoid'))

#model.add(Dense(64, activation='relu', input_dim=20))
#model.add(Dropout(0.5))
#model.add(Dense(64, activation='relu'))
#model.add(Dropout(0.5))
#model.add(Dense(2, activation='softmax'))

sgd = keras.optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
#model.compile(loss=keras.losses.categorical_crossentropy,
#              optimizer=keras.optimizers.SGD(lr=0.01, momentum=0.9, nesterov=True))

model.fit(X, y, epochs=20, batch_size=10)

res = model.evaluate(X, y)
y_pred = model.predict_classes(X)
showData(X, y_pred)








