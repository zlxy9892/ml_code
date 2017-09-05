# -*- coding: utf-8 -*-

from sklearn.ensemble import BaggingClassifier
from sklearn.neighbors import KNeighborsClassifier
clf_bagging = BaggingClassifier(KNeighborsClassifier(), max_samples=0.5, max_features=0.5)

clf_bagging.predict()