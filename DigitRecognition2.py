import numpy as np

from keras.datasets import mnist, fashion_mnist

(trainX, trainY), (testX, testY) = mnist.load_data()

weights = np.zeros((28, 28, 10))
LEARNING_RATE = 0.01

# training
for i in range(60000):
    selection = np.clip(0, 1, trainX[i % 60000])
    actual = trainY[i % 60000]
    expected = np.zeros(10)
    expected[actual] = 1

    output = np.tensordot(selection, weights, axes=([0,1],[0,1]))
    output = 1 / (1 + np.exp(-output))
    #prediction = np.argmax(output)

    errors = output - expected
    weights -= np.outer(selection.flatten(), errors).reshape(28, 28, 10) * LEARNING_RATE

    if i % 10000 == 0:
        print(f"Training... ({i/60000*100:.2f}% Complete)")

# testing
correct = 0
wrong = 0
for i in range(10000):
    selection = np.clip(0, 1, testX[i])
    actual = trainY[i]
    output = np.tensordot(selection, weights, axes=([0,1],[0,1]))
    prediction = np.argmax(output)
    if prediction == testY[i]:
        correct += 1
    else:
        wrong += 1
        #print(np.argmax(output), testY[i], "|", i)
print(f"{correct / 10000*100:.2f}% Accuracy")