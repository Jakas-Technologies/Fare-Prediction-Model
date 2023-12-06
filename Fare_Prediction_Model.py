import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import json
import time


file_path = './dataset/simulated_fare_with_one_feature.csv'
data = pd.read_csv(file_path, sep = ";")
data.head()

X = data["Jarak"].values.reshape(-1, 1)
y = data["Tarif"].values.reshape(-1, 1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

mean_X, std_X = np.mean(X_train), np.std(X_train)
mean_y, std_y = np.mean(y_train), np.std(y_train)

X_train = (X_train - mean_X) / std_X
X_test = (X_test - mean_X) / std_X

y_train = (y_train - mean_y) / std_y
y_test = (y_test - mean_y) / std_y

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(1,)),
    tf.keras.layers.Dense(units=1)
])

optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['mse', 'mae'])

model.fit(X_train, y_train, epochs=100, verbose=1)

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error on Test Data: {mse}')

X_test_denormalized = X_test * std_X + mean_X
y_test_denormalized = y_test * std_y + mean_y
y_pred_denormalized = y_pred * std_y + mean_y

plt.scatter(X_test_denormalized, y_test_denormalized, color='blue', label='True values')
plt.scatter(X_test_denormalized, y_pred_denormalized, color='red', label='Predicted values')
plt.title('Linear Regression Prediction')
plt.xlabel('Distance')
plt.ylabel('Fare')
plt.legend()
plt.show()

test_values = np.arange(800, 16000, 1000).reshape(-1, 1)

test_values_normalized = (test_values - mean_X) / std_X

predicted_fares_normalized = model.predict(test_values_normalized)

predicted_fares_denormalized = predicted_fares_normalized * std_y + mean_y

rounded_predicted_fares = np.round(predicted_fares_denormalized / 1000) * 1000

for i in range(len(test_values)):
    print(f'Jarak: {test_values[i, 0]}, Predicted Fare (Rounded): {rounded_predicted_fares[i, 0]}')

"""
saved_model_path = "./{}.h5".format(int(time.time()))

model.save(saved_model_path)

normalization_params = {
    'mean_X': mean_X,
    'std_X': std_X,
    'mean_y': mean_y,
    'std_y': std_y
}

with open('./normalization_params.json', 'w') as json_file:
    json.dump(normalization_params, json_file)

"""