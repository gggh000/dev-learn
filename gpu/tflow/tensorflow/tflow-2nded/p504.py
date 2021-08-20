import tensorflow as tf
import pandas as pd
import matplotlib as plt
import numpy as np
import sys
import time
import re
import helper

from tensorflow import keras
print(tf.__version__)
print(keras.__version__)
DEBUG=0

CONFIG_ENABLE_PLOT=0
CONFIG_SAVE_MODEL=0

DEBUG=0
CONFIG_ENABLE_PLOT=0
CONFIG_EPOCHS=30
CONFIG_BATCH_SIZE=32


if  len(sys.argv) > 1:
    CONFIG_EPOCHS, CONFIG_BATCH_SIZE = helper.process_params(sys.argv, ["epochs", "batch_size"])

def generate_series(batch_size, n_steps):
    freq1, freq2, offsets1, offsets2 = np.random.rand(4, batch_size, 1)
    print("freq1, freq2, offset1, offset2: ", freq1.shape, freq2.shape, offsets1.shape, offsets2.shape)
    time1 = np.linspace(0, 1, n_steps)
    print("time1: ", time1.shape)

    # wave 1

    series = 0.5 * np.sin((time1 - offsets1) * (freq1 * 10 + 10))

    # wave 2

    series += 0.3 * np.sin((time1 - offsets2) * (freq2 * 20 + 20)) 

    # noise

    series += 0.1 * (np.random.rand(batch_size, n_steps) - 0.5) 
    print("returning series (shape): ", series.shape)
    return series[..., np.newaxis].astype(np.float32)

n_steps=50
series= generate_series(10000, n_steps)
X_train, y_train = series[:7000, :n_steps], series[:7000, -1]
X_valid, y_valid = series[7000:9000, :n_steps], series[7000:9000, -1]
X_test, y_test = series[9000:, :n_steps], series[9000:, -1]

print("X_train, y_train shape: ", X_train.shape, y_train.shape)
print("X_valid, y_valid shape: ", X_valid.shape, y_valid.shape)
print("X_test, y_test shape:   ", X_test.shape, y_test.shape)

if CONFIG_ENABLE_PLOT:
    '''
    fig, ax = plt.subplots()
    ax.plot(series)

    plt.show()
    '''

y_pred = X_valid[:, -1]
print("y_pred shape: ", y_pred.shape)
print("test with basic pred: ", np.mean(keras.losses.mean_squared_error(y_valid, y_pred)))

