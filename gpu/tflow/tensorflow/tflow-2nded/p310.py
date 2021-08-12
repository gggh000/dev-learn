# using neural net for regression task. 
# This is similar to p308.py except p308.py uses fully connected sequential neural net whereas
# this one uses nonsequential network. p309.py's data does not use the non-sequential network's 
# advanage whereas in this example, it does by splitting data into separate training path.

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow import keras

housing = fetch_california_housing()

X_train_full, X_test, y_train_full, y_test = train_test_split(housing.data, housing.target)
X_train, X_valid, y_train, y_valid = train_test_split(X_train_full, y_train_full)

print("X_train/y_train/full/test shapes: ", X_train_full.shape, X_test.shape, y_train_full.shape, y_test.shape)
print("X_/y_/train/valid: ", X_train.shape, X_valid.shape, y_train.shape, y_valid.shape)

X_train_A, X_train_B = X_train[:, :5], X_train[:, 2:]
X_valid_A, X_valid_B = X_valid[:, :5], X_valid[:, 2:]
X_test_A, X_test_B = X_test[:, :5], X_test[:, 2:]
X_new_A, X_new_B = X_test_A[:3], X_test[:3, 2:]

print("X_train_A/B/valid_A/B: ", X_train_A.shape, X_train_B.shape, X_valid_A.shape, X_valid_B.shape)
print("X_test_A/B/new_A/B: ", X_test_A.shape, X_test_B.shape, X_new_A.shape, X_new_B.shape)

scaler=StandardScaler()
X_train = scaler.fit_transform(X_train)
X_valid = scaler.transform(X_valid)
X_test = scaler.transform(X_test)

input_A = keras.layers.Input(shape=[5] , name="wide_input")
input_B = keras.layers.Input(shape=[6] , name="deep_input")
hidden1 = keras.layers.Dense(30, activation="relu")(input_B)
hidden2 = keras.layers.Dense(30, activation="relu")(hidden1)
concat = keras.layers.Concatenate()([input_A, hidden2])
output = keras.layers.Dense(1, name="output")(concat)
model = keras.Model(inputs=[input_A, input_B], outputs=[output])

model.compile(loss="mse", optimizer=keras.optimizers.SGD(lr=1e-3))

history = model.fit( \
    (X_train_A, X_train_B), y_train, \
    epochs=20, \
    validation_data=((X_valid_A, X_valid_B),y_valid)\
    )
print("training result (shape): ", history)
mse_test = model.evaluate((X_test_A, X_test_B), y_test)
y_pred = model.predict((\
    X_new_A, \
    X_new_B))

model.save("p310.h5")
