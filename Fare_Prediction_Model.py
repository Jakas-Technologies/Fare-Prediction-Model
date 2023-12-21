import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import json
import time

file_path = './dataset/dataset_fare.csv'
data = pd.read_csv(file_path, sep = ",")
print(data.head())

data['Distance_BBM'] = data['Distance'] * data['BBM']

X = data[["Distance", "Type", "BBM", "Distance_BBM"]]
y = data["Fare"]
X = pd.get_dummies(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

mean_X, std_X = np.mean(X_train, axis=0), np.std(X_train, axis=0)
mean_y, std_y = np.mean(y_train, axis=0), np.std(y_train, axis=0)

X_train = (X_train - mean_X) / std_X
X_test = (X_test - mean_X) / std_X

y_train = (y_train - mean_y) / std_y
y_test = (y_test - mean_y) / std_y

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(X_train.shape[1],)),
    tf.keras.layers.Dense(units=16, activation='relu'),
    tf.keras.layers.Dense(units=32, activation='relu'),
    tf.keras.layers.Dense(units=16, activation='relu'),
    tf.keras.layers.Dense(units=1, activation='linear')
])

optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['mse'])

history = model.fit(X_train, y_train, epochs=100, verbose=1, validation_data=(X_test, y_test))

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error on Test Data: {mse}')

X_test_denormalized = X_test * std_X + mean_X
y_test_denormalized = y_test * std_y + mean_y
y_pred_denormalized = y_pred * std_y + mean_y

distance = 1000
bbm = 10000
distanceBBM = distance * bbm

test_values = pd.DataFrame({
                              'Distance': [distance, distance, distance],
                              'Type_General': [1, 0, 0],
                              'Type_Student': [0, 1, 0],
                              'Type_Elderly': [0, 0, 1],
                              'BBM': [bbm, bbm, bbm],
                              'Distance_BBM': [distanceBBM, distanceBBM, distanceBBM]
                          })

test_values_normalized = (test_values - mean_X) / std_X

predicted_fares_normalized = model.predict(test_values_normalized)

predicted_fares_denormalized = predicted_fares_normalized * std_y + mean_y

for i in range(len(test_values)):
    predicted_fare_denormalized = predicted_fares_denormalized[i][0]
    rounded_predicted_fare = np.round(predicted_fare_denormalized / 1000) * 1000
    fare_type = test_values.iloc[i, 1:-2].idxmax()

    print(f'Jarak: {test_values["Distance"][i]}, Type: {fare_type}, Predicted Fare (Rounded): {rounded_predicted_fare}')


saved_model_path = "./{}.h5".format(int(time.time()))

model.save(saved_model_path)

print(f'mean_X: {mean_X}')
print(f'std_X:  {std_X}')
print(f'mean_y: {mean_y}')
print(f'std_y:  {std_y}')

result_dict = {
    "mean_X": mean_X.to_dict(),
    "std_X": std_X.to_dict(),
    "mean_y": float(mean_y),
    "std_y": float(std_y)
}

# Save to JSON file
with open('norm_params.json', 'w') as json_file:
    json.dump(result_dict, json_file, indent=4)