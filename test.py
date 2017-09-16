# -*- coding: utf-8 -*-

from sklearn import datasets

iris = datasets.load_iris()
digits = datasets.load_digits()
print(iris.data)
print(digits.data)

