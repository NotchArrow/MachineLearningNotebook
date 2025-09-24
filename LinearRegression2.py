import random
import time

import matplotlib.pyplot as plt
import numpy as np

# Data & theoretical
theoryM = random.uniform(-10, 10)
theoryB = random.uniform(-10, 10)

xList = np.linspace(-100, 100, 200)
yList = theoryM * xList + theoryB #+ np.random.uniform(-100, 100, len(xList))


# Settings
TRAINING_TIME = 2
LEARNING_RATE = .0001

M_INIT = 1
B_INIT = 0

# Model
m = M_INIT
b = B_INIT
dataSize = len(xList)

startTime = time.time()
while time.time() - startTime < TRAINING_TIME:
    prediction = m * xList + b
    errors = prediction - yList
    # meanSquaredError = np.mean(errors ** 2)

    derivativeCostM = sum(errors * xList) * (2.0 / dataSize)
    derivativeCostB = sum(errors) * (2.0 / dataSize)

    m -= LEARNING_RATE * derivativeCostM
    b -= LEARNING_RATE * derivativeCostB

print()
print(f"Theory: y = {theoryM}x + {theoryB}")
print(f"Predicted: y = {m}x + {b}")

lineX = np.linspace(min(xList), max(xList), len(xList))
lineY = m * lineX + b
plt.plot(xList, yList, '.')
plt.plot(lineX, lineY)
plt.show()

