# Using neural net to do a classification task.

import tensorflow as tf
import pandas as pd 
import matplotlib as plt
import sys 
import time
import re
import numpy as np
import helper
from tensorflow import keras

DEBUG=0
CONFIG_ENABLE_PLOT=0
CONFIG_EPOCHS=10
CONFIG_BATCH_SIZE=32

for i in sys.argv:
    print("Processing ", i)
    try:
        if re.search("epochs=", i):
            CONFIG_EPOCHS=int(i.split('=')[1])

        if re.search("batch_size=", i):
            CONFIG_BATCH_SIZE=int(i.split('=')[1])

    except Exception as msg:
        print("No argument provided, default values will be used.")

print("epochs: ", CONFIG_EPOCHS)
print("batch_size: ", CONFIG_BATCH_SIZE)

'''
try:
    CONFIG_EPOCHS, CONFIG_BATCH_SIZE = helper.process_params(sys.argv, ["epochs", "batch_size"])
except Exception as msg:
    print(msg)
    print("CONFIG_EPOCHS: ", CONFIG_EPOCHS)
    print("CONFIG_BATCH_SIZE:", CONFIG_BATCH_SIZE)

quit(0)
'''
fashion_mnist = keras.datasets.fashion_mnist
(X_train_full, y_train_full), (X_test, y_test) = fashion_mnist.load_data()
print("X_train_full.shape: ", X_train_full.shape)
print("X_train_full.dtype: ", X_train_full.dtype)

SEPARATOR=10000
X_valid, X_train = X_train_full[:SEPARATOR] / 255.0, X_train_full[SEPARATOR:]/255.0
y_valid, y_train = y_train_full[:SEPARATOR], y_train_full[SEPARATOR:]
X_test = X_test / 255.0
class_names = ["T-shirt/top","Trouser", "Pullover", "Dress", "Coat" , "Sandal", "Shirt", "Sneaker","Bad","Ankle boot"]

model=keras.models.Sequential()
model.add(keras.layers.Flatten(input_shape = [28, 28]))
model.add(keras.layers.Dense(300, activation="relu"))
model.add(keras.layers.Dense(100, activation="relu"))
model.add(keras.layers.Dense(30, activation="softmax"))

print("model summary: ", model.summary())

model.compile(loss="sparse_categorical_crossentropy", optimizer="sgd", metrics=["accuracy"])
history=model.fit(X_train, y_train, epochs=CONFIG_EPOCHS, batch_size=CONFIG_BATCH_SIZE, validation_data=(X_valid, y_valid))

if CONFIG_ENABLE_PLOT:
    pd.DataFrame(history.history).plot(figsize=(8, 5))
    plt.pyplot.grid(True)
    plt.pyplot.gca().set_ylim(0, 1)
    plt.pyplot.show()

model.evaluate(X_test, y_test)

if DEBUG:
    print("model layers: ", model.layers)

weights, biases  = model.layers[1].get_weights()

if DEBUG:
    print("weights, biases (shapes): ", weights, biases, weights.shape, biases.shape) 

model.save("p297.h5")
X_new = X_test[:3]
print("X_new shape: ", X_new.shape)
y_proba = model.predict(X_new)
print("y_proba (predict)(value): ", y_proba.round(2), "\ny_proba(shape)", np.array(y_proba).shape)

y_pred = model.predict_classes(X_new)
print("y_pred (predict_classes): ", y_pred)


