# -*- coding: utf-8 -*-

from sklearn.datasets import load_iris
from sklearn import tree
import matplotlib.pyplot as plt
import numpy as np
import pydotplus


def showData(dataSet, labels):
    plt.scatter(dataSet[:,0], dataSet[:,1], c=labels)
    plt.show()



# ---------------------------- main ---------------------------- #

iris = load_iris()
X = iris.data
Y = iris.target
print iris.feature_names
print iris.target_names


clf = tree.DecisionTreeClassifier()
clf.fit(X, Y)

# generate a PDF file (or any other supported file type) directly in Python
dot_data = tree.export_graphviz(clf, out_file=None)
graph = pydotplus.graph_from_dot_data(dot_data)
#graph.write_pdf("./dt_iris.pdf")
graph.write_jpg("./dt_iris.jpg")

#showData(dataSet, labels)
