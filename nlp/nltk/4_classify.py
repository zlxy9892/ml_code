# -*- coding: utf-8 -*-

import nltk
from nltk.corpus import names
import random

def gender_features(word):
    return {'last_letter': word[-1]}


names = ([(name, 'male') for name in names.words('male.txt')] +
          [(name, 'female') for name in names.words('female.txt')]
        )
random.shuffle(names)

featureSets = [(gender_features(n), g) for (n,g) in names]
train_set, test_set = featureSets[500:], featureSets[:500]
clf = nltk.NaiveBayesClassifier.train(train_set)
pred_gender = clf.classify(gender_features('Neo'))
print(pred_gender)
