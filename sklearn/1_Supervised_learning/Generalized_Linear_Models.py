# -*- coding: utf-8 -*-

from sklearn import datasets, linear_model
import numpy as np
import matplotlib.pyplot as plt

# 读取糖尿病数据
diebates = datasets.load_diabetes()

# 选取相应的X，y
X = diebates.data[:, np.newaxis, 2]
y = diebates.target

# 划分训练集与测试集
X_train = X[:-20]
y_train = y[:-20]
X_test = X[-20:]
y_test = y[-20:]

# 训练线性回归模型
regr = linear_model.LinearRegression()
regr.fit(X_train, y_train)
print regr

# 输出回归系数(也可看作为回归线的斜率)
print 'Coefficient: ', regr.coef_

# 输出MSR (mean squared error)
print 'Mean squared error: ', np.mean( (regr.predict(X_train) - y_train)**2 )

# Explain variance score: 1 is perfect prediction
print 'Variance score: ', regr.score(X_test, y_test)


# 绘制结果
plt.scatter(X, y, c='black')
plt.plot(X, regr.predict(X), c='blue', linewidth=1)

plt.show()














