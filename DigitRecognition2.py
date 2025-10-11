import numpy as np

from keras.datasets import mnist, fashion_mnist

(trainX, trainY), (testX, testY) = mnist.load_data()

weights = np.zeros((28, 28, 10))
new_weights = weights.copy()

for j in range(3):
    correct = 0
    wrong = 0
    for iteration in range(1000):
        selection = trainX[iteration]
        for x in range(28):
            for y in range(28):
                if selection[x][y] > 0:
                    selection[x][y] = 1

        #print(trainY[iteration])
        output = [0] * 10
        for x in range(28):
            for y in range(28):
                for z in range(10):
                    output[z] += weights[x,y, z] * selection[x, y]

        #print(output)
        prediction = np.argmax(output)
        #print(prediction)
        if prediction == trainY[iteration]:
            correct += 1
        else:
            wrong += 1

        expected = [0] * 10
        expected[trainY[iteration]] = 1
        #print(expected)
        output = np.array(output)
        expected = np.array(expected)
        errors = output - expected
        #print(errors)
        for i, error in enumerate(list(errors)):
            for x in range(28):
                for y in range(28):
                    new_weights[x,y, i] -= error * selection[x, y] * 0.00001
                    #if error != 0 and selection[x][y] > 0:
                        #print(weights[x,y, i])
    print(trainY[iteration])
    print(output)
    print(correct, wrong)
    weights = new_weights.copy()

correct = 0
wrong = 0
for i in range(100):
    selection = testX[i]
    output = [0] * 10
    for x in range(28):
        for y in range(28):
            for z in range(10):
                output[z] += weights[x, y, z] * selection[x, y]
    prediction = np.argmax(output)
    if prediction == testY[i]:
        correct += 1
    else:
        wrong += 1
    print(np.argmax(output), testY[i])
print(correct, wrong)