# -*- coding: utf-8 -*-
"""
Created on Thu Nov 22 19:34:14 2018

@author: Zaman Wahid
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier 
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier

#data preprocessing
dataset = pd.read_csv("dataset.csv")
feature = ['reason_for_absence','transportation_expense','distance_from_residence_to_work', 'service_time','age', 'hit_target','son','body_mass_index']
x = dataset[feature]
y = dataset.absent_class


#Dataset splitting
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=1234, test_size = 0.2)

#Training Model
#Decision Tree
dtc = DecisionTreeClassifier(criterion = 'gini', max_leaf_nodes = 5, random_state=1)
dtc.fit(x_train, y_train)

#Random Forest
rf = RandomForestClassifier(criterion='gini', max_leaf_nodes =25, random_state=1)
rf.fit(x_train, y_train)

#Gradient Boosting
gb = GradientBoostingClassifier(learning_rate = 0.01, max_leaf_nodes=5, random_state=1)
gb.fit(x_train, y_train)

#Prediction
dtc_predict = dtc.predict(x_test)
rf_predict = rf.predict(x_test)
gb_predict = gb.predict(x_test)


#Accuracy Score
from sklearn.metrics import accuracy_score
print("Decision Tree: ",accuracy_score(y_test, dtc_predict)*100)
print("Random Forest: ",accuracy_score(y_test, rf_predict)*100)
print("Gradient Boosted Tree: ", accuracy_score(y_test, gb_predict)*100)