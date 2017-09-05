# -*- coding: utf-8 -*-

from sklearn import datasets, naive_bayes

iris = datasets.load_iris()
X = iris.data
y = iris.target

gnb = naive_bayes.GaussianNB()
gnb.fit(X, y)

y_pred = gnb.predict(X)
print gnb.score(X, y)