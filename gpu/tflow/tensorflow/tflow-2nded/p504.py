import tensorflow as tf
import pandas as pd
import matplotlib as plt
import numpy as np

from tensorflow import keras
print(tf.__version__)
print(keras.__version__)

CONFIG_ENABLE_PLOT=0
CONFIG_SAVE_MODEL=0
def generate_series(batch_size, n_steps):
    freq1, freq2, offsets1, offsets2 = np.random.rand(4, batch_size, 1)
    print(freq1.shape, freq2.shape, offsets1.shape, offsets2.shape)
    time = np.linspace(0, 1, n_steps)

    # wave 1

    series = 0.5 * np.sin((time - offsets1) * (freq1 * 10 + 10))

    # wave 2

    series += 0.3 * np.sin((time - offsets2) * (freq2 * 20 + 20)) 

    # noise

    series += 0.1 * (np.random.rand(batch_size, n_steps) - 0.5) 
    return series[..., np.newaxis].astype(np.float32)

n_steps=50
series= generate_series(10000, n_steps)
X_train, y_train = series[:7000, :n_steps], series[:7000, -1]
X_valid, y_valid = series[7000:9000, :n_steps], series[7000:9000, -1]
X_test, y_test = series[9000:, :n_steps], series[9000:, -1]

print(series.shape)
print("X_train, y_train shape: ", X_train.shape, y_train.shape)
print("X_valid, y_valid shape: ", X_valid.shape, y_valid.shape)
print("X_test, y_test shape:   ", X_test.shape, y_test.shape)
#print(series)
if CONFIG_ENABLE_PLOT:
    '''
    fig, ax = plt.subplots()
    ax.plot(series)

    plt.show()
    '''

y_pred = X_valid[:, -1]
print("y_pred shape: ", y_pred.shape)
#print("test with basic pred: ", np.mean(keras.losses.mean_squared_error(y_valid, y_pred)))

model = keras.models.Sequential([ \
    keras.layers.Flatten(input_shape= [ 50, 1 ]), \
    keras.layers.Dense(1) \
])

model.compile(loss="sparse_categorical_crossentropy", optimizer="sgd", metrics=["accuracy"])
history=model.fit(X_train, y_train, epochs=20, validation_data=(X_valid, y_valid))

if CONFIG_ENABLE_PLOT:
    pd.DataFrame(history.history).plot(figsize=(8, 5))
    plt.pyplot.grid(True)
    plt.pyplot.gca().set_ylim(0, 1)
    plt.pyplot.show()

model.evaluate(X_test, y_test)

print("model layers: ", model.layers)
weights, biases  = model.layers[1].get_weights()
print("weights, biases (shapes): ", weights, biases, weights.shape, biases.shape)

if CONFIG_SAVE_MODEL:
    model.save("p504.h5")

X_new = X_test[:3]
y_proba = model.predict(X_new)
print("y_pred: ", y_pred)


