import random
import time

import matplotlib.pyplot as plt
import numpy as np

dataSize = 5
var1 = np.array([1, 2, 3, 4, 5])
var2 = np.array([2, 4, 6, 8, 10])
w1 = 1
w2 = 41
b = 500
yList = b + var1 * w1 + var2 * w2


# Settings
TRAINING_TIME = 2
LEARNING_RATE = .0001

w1Prediction = 0
w2Prediction = 0
bPrediction = 0

startTime = time.time()
while time.time() - startTime < TRAINING_TIME:
    prediction = w1Prediction * var1 + w2Prediction * var2 + bPrediction
    errors = prediction - yList

    derivativeCostW1 = sum(errors * var1) * (2 / dataSize)
    derivativeCostW2 = sum(errors * var2) * (2 / dataSize)
    derivativeCostB = sum(errors) * (2 / dataSize)

    w1Prediction -= LEARNING_RATE * derivativeCostW1
    w2Prediction -= LEARNING_RATE * derivativeCostW2
    bPrediction -= LEARNING_RATE * derivativeCostB

print()
print(f"Theory: y = {b} + {w1}var1 + {w2}var2")
print(f"Predicted: y = {bPrediction} + {w1Prediction}var1 + {w2Prediction}var2")
print(f"Theory: {yList}")
print(f"Predicted: {b + w1Prediction *var1 + w2Prediction * var2}")