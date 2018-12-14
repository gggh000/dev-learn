#!/usr/bin/python
from sklearn.datasets import load_iris
iris_dataset = load_iris()

print iris_dataset
print("Type of data: {}".format(type(iris_dataset['data'])))
print("Shape of data: {}".format(iris_dataset['data'].shape))
print("1st 5 rows of data: \n{}".format(iris_dataset['data'][:5]))

print("Type of target: \n{}".format(type(iris_dataset['target'])))
print("Shape of target: \n{}".format(iris_dataset['target'].shape))
print("Target:\n{}".format(iris_dataset['target']))

