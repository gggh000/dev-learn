import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
import sys
sys.path.append("/git.co/handson-ml") 
print(sys.path)

#import import_ipynb
from one import *

PATH_RESOURCE="/git.co/handson-ml/datasets/lifesat/"
# Load the data
country_stats = pd.read_csv(PATH_RESOURCE + "oecd_bli_2015.csv", thousands=',')
gpd_per_capita = pd.read_csv(PATH_RESOURCE + "gdp_per_capita.csv", thousands=',', delimiter='\t', encoding='latin1', na_values="n\a")

# prepare the data
country_stats = prepare_country_stats(oecd_bli, gdp_per_capita)
X = np.c_[country_stats["GDP per capita"]]
y = np.c_[country_stats["Life satisfaction"]]

# Visualize the data
country_stats.plot(kind='scatter',  x="GDP per capita", y='Life satisfaction')
plt.show()

# select a linear model
model = sklearn.linear_model.LinearRegression()

# Train the model.
model.fit(X, y)

# Make a prediction for Cyprus

X_new = [[22587]] # cypris'GDP per capita
print(model.predict(X_new)) #outputs.

