# p262

from sklearn.svm import SVC
from sklearn.datasets import load_iris
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
import numpy as np
import mglearn
iris = load_iris()

'''
# split data into train+validation set and test set

X_trainval, X_test, y_trainval, y_test = train_test_split(iris.data, iris.target, random_state=0)

# split train+validation set into training and validation sets

X_train, X_valid, y_train, y_valid = train_test_split(X_trainval, y_trainval, random_state=1)

print("Size of training set: {} size of validation set: {} size of test set:"
 " {}\n".format(X_train.shape[0], X_valid.shape[0], X_test.shape[0]))
 
best_score = 0
for gamma in [0.001, 0.01, 0.1, 1, 10, 100]:
	for C in [0.001, 0.01, 0.1, 1, 10, 100]:

		# for each combination of parameters, train an SVC
	
		print "gamma/C:", gamma, C
		svm = SVC(gamma=gamma, C=C)
		svm.fit(X_train, y_train)
	 
		# evaluate the SVC on the test set
		
		score = svm.score(X_valid, y_valid)
	 
		# if we got a better score, store the score and parameters
		
		if score > best_score:
			best_score = score
			best_parameters = {'C': C, 'gamma': gamma}
			
		# rebuild a model on the combined training and validation set,
		# and evaluate it on the test set
	
svm = SVC(**best_parameters)
svm.fit(X_trainval, y_trainval)
test_score = svm.score(X_test, y_test)
print("Best score on validation set: {:.2f}".format(best_score))
print("Best parameters: ", best_parameters)
print("Test set score with best parameters: {:.2f}".format(test_score))


for gamma in [0.001, 0.01, 0.1, 1, 10, 100]:
	for C in [0.001, 0.01, 0.1, 1, 10, 100]:
	
		# for each combination of parameters,
		# train an SVC
		
		svm = SVC(gamma=gamma, C=C)
		
		# perform cross-validation
		
		scores = cross_val_score(svm, X_trainval, y_trainval, cv=5)
		
		# compute mean cross-validation accuracy
		
		score = np.mean(scores)
		
		# if we got a better score, store the score and parameters
		
		if score > best_score:
			best_score = score
			best_parameters = {'C': C, 'gamma': gamma}
			# rebuild a model on the combined training and validation set
			
svm = SVC(**best_parameters)
svm.fit(X_trainval, y_trainval)
mglearn.plots.plot_cross_val_selection()
'''

#	Using GridSearchCV instead of nested loop above to do a grid search
#	with cross validation.

import pandas as pd
from IPython.display import display

param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100],
 'gamma': [0.001, 0.01, 0.1, 1, 10, 100]}
print("Parameter grid:\n{}".format(param_grid))

from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
grid_search = GridSearchCV(SVC(), param_grid, cv=5)
print grid_search

X_train, X_test, y_train, y_test = train_test_split(
 iris.data, iris.target, random_state=0)

grid_search.fit(X_train, y_train)
print("Test set score: {:.2f}".format(grid_search.score(X_test, y_test)))
print("Best parameters: {}".format(grid_search.best_params_))
print("Best cross-validation score: {:.2f}".format(grid_search.best_score_))
print("Best estimator:\n{}".format(grid_search.best_estimator_))

# convert to DataFrame
results = pd.DataFrame(grid_search.cv_results_)
# show the first 5 rows
display(results.head())


scores = np.array(results.mean_test_score).reshape(6, 6)
# plot the mean cross-validation scores
mglearn.tools.heatmap(scores, xlabel='gamma', xticklabels=param_grid['gamma'],
 ylabel='C', yticklabels=param_grid['C'], cmap="viridis")
 
 