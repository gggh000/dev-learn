import os
import tarfile
import pandas as pd
import urllib.request
import matplotlib.pyplot as plt
import numpy as np
ENABLE_PLOT=0
DOWNLOAD_ROOT="http://raw.githubusercontent.com/ageron/handson-ml2/master/"
HOUSING_PATH=os.path.join("datasets", "housing")
HOUSING_URL=DOWNLOAD_ROOT + "datasets/housing/housing.tgz"

def fetch_housing_data(housing_url=HOUSING_URL, housing_path=HOUSING_PATH):
    os.makedirs(housing_path, exist_ok=True)
    tgz_path=os.path.join(housing_path, "housing.tgz")
    urllib.request.urlretrieve(housing_url, tgz_path)
    housing_tgz=tarfile.open(tgz_path)
    housing_tgz.extractall(path=housing_path)
    housing_tgz.close()

#p52
def split_train_test(data, test_ratio):
    DEBUG = 1
    shuffled_indices = np.random.permutation(len(data))

    if DEBUG:
        print("shuffled_indices: ", shuffled_indices)

    test_set_size = int(len(data) * test_ratio)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    return data.iloc[train_indices], data.iloc[test_indices]    


#p52-end
print("housing: ", HOUSING_PATH)
fetch_housing_data()

def load_housing_data(housing_path=HOUSING_PATH):
    csv_path = os.path.join(housing_path, "housing.csv")
    return pd.read_csv(csv_path)

housing=load_housing_data()

#p48
print(housing.info())
print(housing["ocean_proximity"].value_counts())
print(housing.describe())

if ENABLE_PLOT:
    housing.hist(bins=50, figsize=(20,15))
    plt.show()

train_set, test_set=split_train_test(housing, 0.2)
print(len(train_set), len(test_set))
