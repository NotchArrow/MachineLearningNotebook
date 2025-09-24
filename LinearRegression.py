import matplotlib.pyplot as plt
import numpy as np
import time


# Data & theoretical
THEORETICAL_VALUES = True
theoryM = 2
theoryB = 1

xList = []
yList = []
for i in range(-50, 100):
    xList.append(i)
    yList.append(i * theoryM + theoryB)
dataSize = len(xList)


# Settings
mScale = 1
bScale = 1

TRAINING_TIME = 2
HYPERPARAMETER_ITERATIONS = 5
HYPERPARAMETER_ALLOTMENT = .1

LEARNING_RATE = .01
SCALAR_LEARNING_RATE = 0.05

M_INIT = 1
B_INIT = 0

# Model
m = M_INIT
b = B_INIT
trainingStartTime = time.time()
while time.time() - trainingStartTime < TRAINING_TIME:

    # Hyperparameter tuning
    while time.time() - trainingStartTime < TRAINING_TIME * HYPERPARAMETER_ALLOTMENT and THEORETICAL_VALUES:
        iterationStartTime = time.time()
        while time.time() - iterationStartTime < TRAINING_TIME * HYPERPARAMETER_ALLOTMENT / HYPERPARAMETER_ITERATIONS:
            mChange = 0
            bChange = 0
            for i in range(dataSize):
                x = xList[i]
                y = yList[i]
                predictedY = m * x + b
                error = predictedY - y
                mChange += 2 * error * x
                bChange += 2 * error

            m -= LEARNING_RATE * (mChange / dataSize ** 2) * mScale
            b -= LEARNING_RATE * (bChange / dataSize ** 2) * bScale

        mError = m - theoryM
        print(f"M Scalar: {mScale} -> {mScale - mError * SCALAR_LEARNING_RATE}")
        mScale -= mError * SCALAR_LEARNING_RATE
        bError = b - theoryB
        print(f"B Scalar: {bScale} -> {bScale - bError * SCALAR_LEARNING_RATE}")
        bScale -= bError * SCALAR_LEARNING_RATE

    # Training for remaining time
    mChange = 0
    bChange = 0
    for i in range(dataSize):
        x = xList[i]
        y = yList[i]
        predictedY = m * x + b
        error = predictedY - y
        mChange += 2 * error * x
        bChange += 2 * error

    m -= LEARNING_RATE * (mChange / dataSize ** 2) * mScale
    b -= LEARNING_RATE * (bChange / dataSize ** 2) * bScale

print()
print(f"y = {m}x + {b}")

lineX = np.linspace(min(xList), max(xList), len(xList))
lineY = m * lineX + b
plt.plot(lineX, lineY)
plt.plot(xList, yList, '.')
plt.show()
