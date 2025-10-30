import numpy as np

from keras.datasets import mnist

(trainX, trainY), (testX, testY) = mnist.load_data()

rng = np.random.default_rng()

weights_input_output = rng.normal(0, 0.1, (10, 28 ** 2))
#weights_input_output = np.zeros((10, 28 ** 2))
LEARNING_RATE = 0.1
EPOCHS = 5

for epoch in range(EPOCHS):
    for data in range(60000):
        input_data = trainX[data].flatten() / 255.0
        #print(input_data.shape)

        actual = trainY[data]
        expected = np.zeros(10)
        expected[actual] = 1
        #print(expected)

        output = (input_data * weights_input_output).sum(1)
        #print(output)

        output_final = 1 / (1 + np.exp(-output))
        #print(output_final)

        error = output_final - expected
        #print(error)

        weights_input_output -= np.outer(error, input_data) * LEARNING_RATE

        #break
        if data % 10000 == 0:
            print(f"Epoch: {epoch}/{EPOCHS} ({((epoch/EPOCHS) + (data/60000/EPOCHS))*100:.2f}% Complete).")

    LEARNING_RATE /= 10

# testing
correct = 0
wrong = 0
for data in range(10000):
    input_data = testX[data].flatten() / 255.0
    actual = testY[data]

    output = (input_data * weights_input_output).sum(1)

    output_final = 1 / (1 + np.exp(-output))

    prediction = np.argmax(output_final)
    if prediction == actual:
        correct += 1
    else:
        wrong += 1
        #print(np.argmax(output), testY[i], "|", i)
print("==========")
print(f"{correct / 10000*100:.2f}% Accuracy")