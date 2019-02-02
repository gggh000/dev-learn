#!/usr/bin/python

def print_k(pStr):
	print("-------------------------------------------------------------")
	print(str(pStr))
	print("-------------------------------------------------------------")
	
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier 

import pandas as pd
import numpy as np
import mglearn 
import matplotlib.pyplot as plt

iris_dataset = load_iris()
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(iris_dataset['data'], iris_dataset['target'], random_state=0)

print iris_dataset
print("Type of data: {}".format(type(iris_dataset['data'])))
print("Shape of data: {}".format(iris_dataset['data'].shape))
print("1st 5 rows of data: \n{}".format(iris_dataset['data'][:5]))

print("Type of target: \n{}".format(type(iris_dataset['target'])))
print("Shape of target: \n{}".format(iris_dataset['target'].shape))
print("Target:\n{}".format(iris_dataset['target']))

iris_dataframe=pd.DataFrame(X_train, columns=iris_dataset.feature_names)
print("X_train shape: {}".format(X_train.shape))
print("y_train shape: {}".format(y_train.shape))
print("X_test shape: {}".format(X_test.shape))
print("y_test shape: {}".format(y_test.shape))

print_k("(OPTIONAL): SHOW PLOT.")
pd.plotting.scatter_matrix(iris_dataframe, c=y_train, figsize=(15,15), marker='o', hist_kwds={'bins': 20}, s=60, alpha=.8, cmap=mglearn.cm3)
#plt.show()

print_k("CREATE PREDICTION OBJECT BASED ON K-NEIGHTBOUR CLASSIFIER, KNN OBJECT.")
knn = KNeighborsClassifier(n_neighbors=1) 
knn.fit(X_train, y_train)

print_k("PREDICTION FOR SINGLE DATA: X_new")
X_new = np.array([[5, 2.9, 1, 0.2]]) 
print("X_new.shape: {}".format(X_new.shape))
prediction = knn.predict(X_new) 
print("Prediction: {}".format(prediction)) 
print("Predicted target name: {}".format(       iris_dataset['target_names'][prediction]))

print_k("PREDICTION FOR X_TEST DATA SET")
y_pred = knn.predict(X_test) 
print("Test set predictions:\n {}".format(y_pred))

print_k("PREDICTION ACCURACY SCORE:")
print("Test set score: {:.2f}".format(np.mean(y_pred == y_test)))

