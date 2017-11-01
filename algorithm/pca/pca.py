# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def loadDataSet(filename):
    df = pd.read_table(filename, sep='\t')
    return np.array(df)

def showData(dataMat, reconMat):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(dataMat[:, 0], dataMat[:, 1], c='green')
    ax.scatter(np.array(reconMat[:, 0]), reconMat[:, 1], c='red')
    plt.show()

def pca(dataMat, topNfeat=999999):

    # 1.对所有样本进行中心化（所有样本属性减去属性的平均值）
    meanVals = np.mean(dataMat, axis=0)
    meanRemoved = dataMat - meanVals

    # 2.计算样本的协方差矩阵 XXT
    covmat = np.cov(meanRemoved, rowvar=0)
    print(covmat)

    # 3.对协方差矩阵做特征值分解，求得其特征值和特征向量，并将特征值从大到小排序，筛选出前topNfeat个
    eigVals, eigVects = np.linalg.eig(np.mat(covmat))
    eigValInd = np.argsort(eigVals)
    eigValInd = eigValInd[:-(topNfeat+1):-1]    # 取前topNfeat大的特征值的索引
    redEigVects = eigVects[:, eigValInd]        # 取前topNfeat大的特征值所对应的特征向量

    # 4.将数据转换到新的低维空间中
    lowDDataMat = meanRemoved * redEigVects     # 降维之后的数据
    reconMat = (lowDDataMat * redEigVects.T) + meanVals # 重构数据，可在原数据维度下进行对比查看
    return np.array(lowDDataMat), np.array(reconMat)


# ---------------------------- main ---------------------------- #

dataMat = loadDataSet('./data/testSet.txt')
lowDDataMat, reconMat = pca(dataMat, 1)
#showData(dataMat, lowDDataMat)
showData(dataMat, reconMat)
print(lowDDataMat)
