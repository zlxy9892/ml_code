# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt


def loadDataSet(filename):
    dataSet = []
    fr = open(filename, 'r')
    for line in fr.readlines():
        lineArr = line.strip().split('\t')
        fltLine = map(float, lineArr)
        dataSet.append(fltLine)
    return dataSet

def showData(dataMat, centroids):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(dataMat[:, 0], dataMat[:, 1], c='green', marker='o', s=30)
    ax.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='+', s=100)
    plt.show()

# 计算两点间的欧几里得距离
def distEuclid(vecA, vecB):
    dist = np.sqrt(np.sum(np.power(vecA - vecB, 2)))
    return dist

# 生成k个随机中心点
def randCenter(dataMat, k):
    m = np.shape(dataMat)[1]
    centroid = np.mat(np.zeros((k, m)))
    for i in range(m):
        minV_i = np.min(dataMat[:, i])
        maxV_i = np.max(dataMat[:, i])
        range_i = maxV_i - minV_i
        rand_i = minV_i + range_i * np.random.rand(k, 1)
        centroid[:, i] = rand_i
    return centroid

# k-均值聚类算法主函数
def kmeans(dataMat, k, distMeasure=distEuclid, createCent=randCenter):
    n = np.shape(dataMat)[0]
    clusterAssment = np.mat(np.zeros((n, 2)))       # 记录每个点的最近中心点和与最近中心点之间的距离
    centroids = createCent(dataMat, k)
    clusterChanged = True
    while clusterChanged:   # 当中心点不再变化之后停止迭代
        clusterChanged = False
        for i in range(n):  # 计算每个点的最近中心点
            minDist = np.inf; minIndex = -1
            for j in range(k):
                distJI = distMeasure(centroids[j, :], dataMat[i, :])
                if minDist > distJI:
                    minDist = distJI
                    minIndex = j
            if clusterAssment[i, 0] != minIndex:
                clusterChanged = True
            clusterAssment[i, :] = minIndex, minDist**2
        print(centroids)
        for cent in range(k):  # 更新中心点
            ix = np.nonzero(clusterAssment[:, 0].A == cent)[0]
            ptsInClust = dataMat[ix]
            centroids[cent, :] = np.mean(ptsInClust, axis=0)    # 若不加axis=0，则默认会将矩阵中所有元素做flatten操作之后计算均值
    return centroids, clusterAssment


# ---------------------------- main ---------------------------- #

dataSet = loadDataSet('./data/testSet.txt')
dataMat = np.mat(dataSet)

k = 4
centroids, clustAssing = kmeans(dataMat, k)
showData(dataMat, centroids)
