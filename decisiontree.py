# -*- coding: utf-8 -*-
"""
Created on Tue Nov 20 20:20:52 2018

@author: Zaman Wahid
"""
import numpy as np
import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn import tree
from sklearn import ensemble

dataset = pd.read_csv("dataset.csv")

X = dataset.values[:, 0:20]
Y = dataset.values[:, 21]

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 1)




clf_gini = DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=5,
            max_features=None, max_leaf_nodes=None, min_samples_leaf=8,
            min_samples_split=2, min_weight_fraction_leaf=0.1,
            presort=False, random_state=80, splitter='best')
clf_gini.fit(X_train, Y_train)


clf_entropy = DecisionTreeClassifier(criterion = "entropy", random_state = 100)
clf_entropy.fit(X_train, Y_train)

clf = ensemble.RandomForestClassifier(criterion = "gini", random_state = 100)
clf.fit(X_train, Y_train)

cclf = ensemble.GradientBoostingClassifier(learning_rate=0.1, random_state = 100, max_leaf_nodes=None)
cclf.fit(X_train, Y_train)

Y_pred = clf_gini.predict(X_test)
y_pred = clf_entropy.predict(X_test)
yy_pred = clf.predict(X_test)
yyy_pred = cclf.predict(X_test)

giniAcc = accuracy_score(Y_test, Y_pred)*100
igAcc = accuracy_score(Y_test, y_pred)*100
ig = accuracy_score(Y_test, yy_pred)*100
ifg = accuracy_score(Y_test, yyy_pred)*100



