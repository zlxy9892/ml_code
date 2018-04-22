# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt


def loadDataSet(filename):
    numFeat = len(open(filename).readline().split('\t')) - 1
    fr = open(filename)
    dataMat = []
    labelMat = []
    for line in fr.readlines():
        lineArr = []
        lineStrs = line.strip().split('\t')
        for i in range(numFeat):
            lineArr.append(float(lineStrs[i]))
        dataMat.append(lineArr)
        labelMat.append(float(lineStrs[-1]))
    fr.close()
    return dataMat, labelMat

# 普通最小二乘，计算w（线性回归中的参数矩阵）
def standRegression(xArr, yArr):
    xMat = np.mat(xArr)
    yMat = np.mat(yArr).T
    xTx = xMat.T * xMat
    if np.linalg.det(xTx) == 0.0:
        print("this matrix is singular, cannot do inverse")
        return
    ws = xTx.I * (xMat.T * yMat)
    return ws

# 绘制数据
def showData(xMat, yMat, ws):
    xMat = np.array(xMat)
    yMat = np.array(yMat)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xMat[:, 1], yMat, marker='.')
    xCopy = xMat.copy()
    xCopy.sort(0)
    yHat = xCopy * ws
    ax.plot(xCopy[:, 1], yHat, c='red')
    plt.show()




# ---------------------------- main ---------------------------- #
xArr, yArr = loadDataSet('./data/ex0.txt')
xMat = np.mat(xArr)
yMat = np.mat(yArr)
ws = standRegression(xMat, yMat)
print(ws)

yHat = xMat * ws
print(np.corrcoef(yHat.T, yMat))

showData(xMat, yMat, ws)
