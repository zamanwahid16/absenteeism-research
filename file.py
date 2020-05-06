# -*- coding: utf-8 -*-
"""
Created on Thu Nov 22 12:47:25 2018

@author: Zaman Wahid
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier

#Importing dataset and selecting the feautures and target variable
dataset = pd.read_csv("dataset.csv")
x = dataset.iloc[:, 0:20].values
y = dataset.absent_class

#Splitting the dataset for the training and testing
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state = 1234, test_size = 0.2)

#Training the dataset: Decision Tree Classifier
dtc = DecisionTreeClassifier(criterion= 'entropy', max_leaf_nodes = 5, random_state=1)
dtc.fit(x_train, y_train)

#Random Forest
rf = RandomForestClassifier(criterion = 'entropy', max_leaf_nodes = 10, random_state=1 )
rf.fit(x_train, y_train)

#Gradient Boosting Trees
gbt = GradientBoostingClassifier(max_leaf_nodes =12, learning_rate=0.01, random_state=1)
gbt.fit(x_train, y_train)

#Prediction
dtc_predict = dtc.predict(x_test)
rf_predict = rf.predict(x_test)
gbt_predict = gbt.predict(x_test)
#accuracy
from sklearn.metrics import accuracy_score
print("Decision Tree: ", accuracy_score(y_test, dtc_predict)*100)
print("Random Forest: ", accuracy_score(y_test, rf_predict)*100)
print("Gradient Boosted Tree: ", accuracy_score(y_test, gbt_predict)*100)