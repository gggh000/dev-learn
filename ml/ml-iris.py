#!/usr/bin/python
from sklearn.datasets import load_iris
import pandas as pd
#import mglearn 
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

#pd.plotting.scatter_matrix(iris_dataframe, c=y_train, figsize=(15,15), marker='o', hist_kwds={'bins': 20}, s=60, alpha=.8, cmap=mglearn.cm3)

