# -*- coding: utf-8 -*-

import math
import operator
import pickle   # 对象持久化


# 构造数据集
def createDataSet():
    dataSet = [
        [1, 1, 'yes'],
        [1, 1, 'yes'],
        [1, 0, 'no'],
        [0, 1, 'no'],
        [0, 1, 'no']
    ]
    labels = ['no surfacing', 'flippers']
    return dataSet, labels

# 计算输入数据集的信息熵
def calcEntropy(dataSet):
    numEntities = len(dataSet)
    labelCounts = {}     # 记录每个标签的数量
    for featVec in dataSet:
        curLabel = featVec[-1]
        if curLabel not in labelCounts.keys():
            labelCounts[curLabel] = 1
        else:
            labelCounts[curLabel] += 1
    entropy = 0.0
    for key in labelCounts:
        prob = float(labelCounts[key]) / numEntities
        entropy += prob * math.log(prob, 2)
    entropy = -1.0 * entropy
    return entropy

# 按照给定特征划分数据集
def splitDataSet(dataSet, axis, value):
    retDataSet = []
    for featVec in dataSet:
        if featVec[axis] == value:
            reducedFeatVec = featVec[:axis]
            reducedFeatVec.extend(featVec[axis+1:])
            retDataSet.append(reducedFeatVec)
    return retDataSet

# 选择最好的数据集划分方式
def chooseBestFeatureToSplit(dataSet):
    numFeatures = len(dataSet[0]) - 1
    baseEntropy = calcEntropy(dataSet)
    bestInfoGain = 0.0
    bestFeature = -1
    for i in range(numFeatures):
        featList = [example[i] for example in dataSet]
        uniqueVals = set(featList)
        newEntropy = 0.0
        for value in uniqueVals:
            subDataSet = splitDataSet(dataSet, i, value)
            prob = len(subDataSet) / float(len(dataSet))
            newEntropy += prob * calcEntropy(subDataSet)    # 计算每种划分方式的信息熵
        infoGain = baseEntropy - newEntropy     # 计算信息增益值
        if bestInfoGain < infoGain:     # 获取最高的信息增益值，以及对应的属性列
            bestInfoGain = infoGain
            bestFeature = i
    return bestFeature

# 返回出现次数最多的分类名称
def majorityCnt(classList):
    classCounts = {}
    for vote in classList:
        if vote not in classCounts.keys():
            classCounts[vote] = 0
        classCounts[vote] += 1
    sortedClassCounts = sorted(classCounts.iteritems(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCounts[0][0]

# 创建决策树
def createTree(dataSet, labels):
    classList = [example[-1] for example in dataSet]
    if classList.count(classList[0]) == len(classList):     # 若所有的类标签完全相同，则直接返回该类标签
        return classList[0]
    if len(dataSet[0]) == 1:        # 若遍历完所有特征，返回出现次数最多的
        return majorityCnt(classList)
    bestFeat = chooseBestFeatureToSplit(dataSet)
    bestFeatLabel = labels[bestFeat]
    myTree = {bestFeatLabel: {}}
    del labels[bestFeat]
    featValues = [example[bestFeat] for example in dataSet]
    uniqueVals = set(featValues)        # 得到列表包含的所有属性值
    for value in uniqueVals:
        subLabels = labels[:]
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat, value), subLabels)
    return myTree

# 使用决策树执行分类
def classify(inputTree, featLabels, testVec):
    firstStr = inputTree.keys()[0]
    secondDict = inputTree[firstStr]
    featIndex = featLabels.index(firstStr)      # 将标签字符串转换为索引
    for key in secondDict.keys():
        if testVec[featIndex] == key:
            if type(secondDict[key]).__name__ == 'dict':
                classLabel = classify(secondDict[key], featLabels, testVec)
            else:
                classLabel = secondDict[key]
    return classLabel

# 使用pickle模块存储决策树
def storeTree(inputTree, filename):
    fw = open(filename, 'w')
    pickle.dump(inputTree, fw)
    fw.close()

# 从存储的持久化文件中读取决策树
def grabTree(filename):
    fr = open(filename)
    return pickle.load(fr)


# ---------------------------- main ---------------------------- #

ds, labels = createDataSet()
print ds
print labels
myTree = createTree(ds, labels)
print myTree
testVec = [1, 1]
ds, labels = createDataSet()
classLabel = classify(myTree, labels, testVec)
print classLabel
storeTree(myTree, './tree.txt')
myTree1 = grabTree('./tree.txt')
print myTree1
