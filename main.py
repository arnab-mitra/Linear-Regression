import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from math import sqrt

# Reading the train.csv dataset
train_data = pd.read_csv('train.csv')

x = train_data['x'].values
y = train_data['y'].values

mean_x = np.mean(x)
mean_y = np.mean(y)

numer = 0
denom = 0
n = len(x)
for i in range(n):
    numer += (x[i] - mean_x) * (y[i] - mean_y)
    denom += (x[i] - mean_x) ** 2

slope = numer / denom
intercept = mean_y - (slope * mean_x)

# Performing predictions on the test data
test_data = pd.read_csv('test.csv')
test_predictions = []
for i in range(len(test_data)):
    y_predicted = intercept + (slope * test_data['x'][i])
    test_predictions.append(y_predicted)

# Calculating RMSE
mse = 0
for i in range(len(test_data)):
    mse += (test_predictions[i] - test_data['y'][i]) ** 2
rmse = sqrt(mse / len(test_data))

print("Test Predictions:")
for i in range(len(test_data)):
    print(f"x={test_data['x'][i]}, y_predicted={test_predictions[i]}")

print(f"RMSE: {rmse}")

# Plotting the actual and predicted values
plt.scatter(test_data['x'], test_data['y'], color='blue', label='Actual')
plt.plot(test_data['x'], test_predictions, color='red', label='Predicted')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Actual vs Predicted')
plt.legend()
plt.show()

