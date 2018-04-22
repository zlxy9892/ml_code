# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt


# CART 的节点
class treeNode():
    def __init__(self, feat, val, left, right):
        featureToSplitOn = feat
        valueOfSplit = val
        leftBranch = left
        rightBranch = right

# 读取数据
def loadDataSet(filename):
    dataMat = []
    fr = open(filename, 'r')
    for line in fr.readlines():
        curLine = line.strip().split('\t')
        fltLine = map(float, curLine)
        dataMat.append(fltLine)
    return dataMat

# 绘制数据
def showData(dataSet):
    dataMat = np.mat(dataSet)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(dataMat[:, 1], dataMat[:, 2])
    plt.show()

# 根据属性和属性值对数据集进行划分（二元切分）
def binSplitDataSet(dataSet, featIndex, value):
    mat0 = dataSet[np.nonzero(dataSet[:, featIndex] <= value)[0], :][0]
    mat1 = dataSet[np.nonzero(dataSet[:, featIndex] > value)[0], :][0]
    return mat0, mat1

def regLeaf(dataSet):
    return np.mean(dataSet[:, -1])

def regErr(dataSet):
    return np.var(dataSet[:, -1]) * np.shape(dataSet)[0]

# 选取最佳的切分方式（返回最佳切分属性和切分的属性值大小)
def chooseBestSplit(dataSet, leafType, errType, ops):
    tolS = ops[0]       # 容许的误差下降值
    tolN = ops[1]       # 切分的最少样本数
    if len(set(dataSet[:, -1].T.tolist()[0])) == 1:
        return None, leafType(dataSet)
    m, n = np.shape(dataSet)
    S = errType(dataSet)
    bestS = np.inf; bestFeatIndex = 0; bestSplitValue = 0
    for featIndex in range(n-1):
        for splitVal in set(dataSet[:, featIndex]):
            mat0, mat1 = binSplitDataSet(dataSet, featIndex, splitVal)
            if (np.shape(mat0)[0] < tolN) or (np.shape(mat1)[0] < tolN):
                continue
            newS = errType(mat0) + errType(mat1)
            if newS < bestS:
                bestFeatIndex = featIndex
                bestSplitValue = splitVal
                bestS = newS
    if (S - bestS) < tolS:
        return None, leafType(dataSet)
    mat0, mat1 = binSplitDataSet(dataSet, bestFeatIndex, bestSplitValue)
    if (np.shape(mat0)[0] < tolN) or (np.shape(mat1)[0] < tolN):
        return None, leafType(dataSet)
    return bestFeatIndex, bestSplitValue

# 构造回归树
def createRegTree(dataSet, leafType=regLeaf, errType=regErr, ops=(1,4)):
    feat, val = chooseBestSplit(dataSet, leafType, errType, ops)
    if feat == None:
        return val
    retTree = {}
    retTree['spInd'] = feat
    retTree['spVal'] = val
    lSet, rSet = binSplitDataSet(dataSet, feat, val)
    retTree['left'] = createRegTree(lSet, leafType, errType, ops)
    retTree['right'] = createRegTree(rSet, leafType, errType, ops)
    return retTree



# ---------------------------- main ---------------------------- #

dataSet = loadDataSet('./data/ex0.txt')
dataSet = np.mat(dataSet)
regTree = createRegTree(dataSet)
print(regTree)
#print dataSet
showData(dataSet)
