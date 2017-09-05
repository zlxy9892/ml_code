# -*- coding: utf-8 -*-

from sklearn import datasets, tree
import matplotlib.pyplot as plt
import numpy as np
import pydotplus

def dt_classification():
    iris = datasets.load_iris()
    X = iris.data[:, 0:2]
    y = iris.target
    
    clf = tree.DecisionTreeClassifier()
    clf.fit(X, y)
    
    dot_data = tree.export_graphviz(clf, out_file=None,
                                    feature_names=iris.feature_names,  
                                    class_names=iris.target_names,  
                                    filled=True, rounded=True,  
                                    special_characters=True
                                    )
    graph = pydotplus.graph_from_dot_data(dot_data)
    graph.write_png("./tree_iris.png")
    
    # plot result
    xmin, xmax = X[:, 0].min() - 1, X[:, 0].max() + 1
    ymin, ymax = X[:, 1].min() - 1, X[:, 1].max() + 1
    plot_step = 0.02
    xx, yy = np.meshgrid(np.arange(xmin, xmax, plot_step),
                         np.arange(ymin, ymax, plot_step))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=plt.cm.Paired)
    
    # Plot the training points
    n_classes = 3
    plot_colors = "bry"
    for i, color in zip(range(n_classes), plot_colors):
        idx = np.where(y == i)
        plt.scatter(X[idx, 0], X[idx, 1], c=color, 
                    label=iris.target_names[i],
                    cmap=plt.cm.Paired)

def dt_regression():
    # create random data set
    rng = np.random.RandomState(0)
    X = np.sort(5 * rng.rand(80, 1))
    y = np.sin(X).ravel()
    y[::5] += 3 * (0.5 - rng.rand(16))  # 每隔5个加上随机数
    
    # fit regression model
    dt_regr = tree.DecisionTreeRegressor(max_depth=2)
    dt_regr.fit(X, y)
    
    # predict
    X_test = np.arange(0.0, 5.0, 0.01)[:, np.newaxis]
    y_test = dt_regr.predict(X_test)
    
    # plot the result
    plt.figure()
    plt.scatter(X, y, c='darkorange', label='data')
    plt.plot(X_test, y_test, c='yellowgreen', label='max_depth=5', linewidth=2)
    plt.xlabel('data')
    plt.ylabel('target')
    plt.title('Decision Tree Regression')
    plt.legend()
    plt.show()

    

# --------------- main ---------------- #
#dt_classification()
dt_regression()



















