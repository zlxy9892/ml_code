# -*- coding: utf-8 -*-

import numpy as np
import operator
import matplotlib
import matplotlib.pyplot as plt


# 产生数据集
def createDataSet():
    group = np.array([ [1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1] ])
    labels = ['A', 'A', 'B', 'B']
    return group, labels

# k邻近算法主函数（knn）
def classify0(inX, dataSet, labels, k):
    # 求inX数据与所有训练数据集的距离
    dataSetSize = dataSet.shape[0]      # 训练数据集的行数（训练样本数量)
    diffMat = np.tile(inX, (dataSetSize, 1)) - dataSet      # tile 函数：复制（m行，n列）的inX
    sqDiffMat = diffMat**2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances**0.5

    # 距离排序
    sortedDistIndicies = distances.argsort()

    # 选择出距离最小的k个点
    classCount = {}     # 记录在距离最近的k个点钟，每个label对应的数量
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1      # Python中字典(Dictionary)中的get() 函数：返回指定键的值，如果值不在字典中返回默认值。

    # 给字典排序
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)     # 根据字典中的第二列的值，即values进行由大到小排序

    return sortedClassCount[0][0]

# 读取约会数据
def file2matrix(filename):
    fr = open(filename)
    arrayOfLines = fr.readlines()
    numberOfLines = len(arrayOfLines)
    resMat = np.zeros((numberOfLines, 3))
    classLabelVector = []
    index = 0
    for line in arrayOfLines:
        line = line.strip()
        listFromLine = line.split('\t')
        resMat[index, :] = listFromLine[0:3]
        classLabelVector.append(int(listFromLine[-1]))
        index += 1
    return resMat, classLabelVector

# 数据可视化
def visualizeData(dataMat, labels):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(dataMat[:, 0], dataMat[:, 1],
               c=15.0 * np.array(labels), s=15.0 * np.array(labels))
    plt.show()

# 数据归一化
def autoNorm(dataSet):
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    normDataSet = np.zeros(np.shape(dataSet))
    m = dataSet.shape[0]
    normDataSet = dataSet - np.tile(minVals, (m, 1))
    normDataSet = normDataSet / np.tile(ranges, (m, 1))
    return normDataSet, ranges, minVals

    print minVals

# 针对约会数据的分类器，得到推测的误差，其中前10%作为测试集，后90%作为训练集
def datingClassTest():
    hoRatio = 0.1
    datingDataMat, datingLabels = file2matrix('./data/datingTestSet2.txt')
    normMat, ranges, minVals = autoNorm(datingDataMat)
    m = normMat.shape[0]
    numTestVecs = int(m * hoRatio)
    errorCount = 0.0
    for i in range(numTestVecs):
        classifierResult = classify0(normMat[i, :], normMat[numTestVecs:m, :], datingLabels[numTestVecs:m], 3)
        print 'the classifer came back with: %d, the real answer is: %d' %(classifierResult, datingLabels[i])
        if classifierResult != datingLabels[i]:
            errorCount += 1.0
    print "the total error rate is: %f" %(errorCount / float(numTestVecs))



# ---------------------------- main ---------------------------- #

#group, labels = createDataSet()
print '------------------ result ------------------'
#res = classify0([0, 0], group, labels, 3)
datingDataMat, datingLabels = file2matrix('./data/datingTestSet2.txt')
print datingDataMat
print datingLabels
visualizeData(datingDataMat, datingLabels)
normMat, ranges, minVals = autoNorm(datingDataMat)
print normMat
datingClassTest()


