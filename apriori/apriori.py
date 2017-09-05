# -*- coding: utf-8 -*-

import itertools

# 生成一组测试数据
def loadDataSet():
    return [ [1, 3, 4], [2, 3, 5], [1, 2, 3, 5], [2, 5] ]
    #return [[1, 2, 5], [2, 4], [2, 3], [1, 2, 4], [1, 3], [2, 3], [1, 3], [1, 2, 3, 5], [1, 2, 3]]


# 生成初始的项集合，每一项作为单个的一个集合，构成集合列表
def createC1(dataset):
    C1 = []
    for transection in dataset:
        for item in transection:
            if not [item] in C1:
                C1.append([item])
    C1.sort()
    return map(frozenset, C1)   # 对C1中每个项构建一个不变集合


# 计算所有项集的支持度
def scanD(D, Ck, minSupport):
    # 生成一个字典，表达每个项集（Ck中的一个元素）在所有数据集合中出现的次数
    ssCnt = {}
    for transection in D:
        for can in Ck:
            if can.issubset(transection):
                if not ssCnt.has_key(can):
                    ssCnt[can] = 1
                else:
                    ssCnt[can] += 1

    # 计算每个项集的支持度
    numItems = float(len(D))
    retList = []        # 满足最小支持度阈值的项集
    supportData = {}    # 支持度数据，key: 项集，value: 支持度
    for key in ssCnt:
        supportValue = ssCnt[key] / numItems
        if supportValue >= minSupport:
            retList.insert(0, key)
        supportData[key] = supportValue

    return retList, supportData


# 将输入的频繁集列表Lk与项集元素个数k，输出为Ck，即输出在Lk的所有元素中，组成k个元素的所有组合情况
# The apriori-gen function takes as argument Lk-1,the set of all large (k - 1)-itemsets. It returns a superset of the set of all large k-items.
# （此处的算法有待商榷）
def aprioriGen(Lk, k):
    retList = []
    lenLk = len(Lk)

    # 第一阶段：自连接
    for i in range(lenLk-1):
        for j in range(i+1, lenLk):
            L1 = list(Lk[i])[:k-2]
            L2 = list(Lk[j])[:k-2]
            L1.sort()
            L2.sort()
            if L1 == L2:
                retList.append(Lk[i] | Lk[j])   # 当前k-2个项相同时，将两个集合合并

    # 第二阶段：剪枝
    rmList = []     # 记录需要删去的超集
    for superset in retList:
        subsets = map(set, itertools.combinations(superset, k-1))
        for subset in subsets:
            if not subset in Lk:
                rmList.append(superset)
                #retList.remove(superset)
                break
    for rmItem in rmList:
        retList.remove(rmItem)

    return retList


# Apriori算法的主函数
def apriori(dataset, minSupport = 0.5):
    C1 = createC1(dataset)
    D = map(set, dataset)
    L1, supportData = scanD(D, C1, minSupport)
    L = [L1]
    k = 2

    while len(L[k-2]) > 0:
        Ck = aprioriGen(L[k-2], k)              # 从项集（每个元素的数量为k-1个）生成新项集（元素数量为k个）
        Lk, supK = scanD(D, Ck, minSupport)
        supportData.update(supK)
        L.append(Lk)

        print 'C' + str(k)
        print Ck
        print 'L' + str(k)
        print Lk

        k += 1

    return L, supportData


# ---------------------------- main ---------------------------- #

ds = loadDataSet()
L, sup = apriori(ds, 0.5)
print '------------------ result ------------------'
print ds
print L
print sup
