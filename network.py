"""Simple neural network to classify MNIST digits - implemented with
stochastic gradient descent, backpropagation, sigmoid neurons and a
quadratic cost function.

Written shortly after reading http://neuralnetworksanddeeplearning.com
so, whilst somewhat different, it's no doubt heavily inspired by that
and much credit due to Michael Nielsen (his code is available at
https://github.com/mnielsen/neural-networks-and-deep-learning).
"""

import random
import timeit

import numpy as np

from data_loaders import load_mnist_data_set


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def sigmoid_deriv(z):
    return sigmoid(z) * (1 - sigmoid(z))


def quadratic_cost_deriv(output_a, y):
    return output_a - y


def randomly_batch(sequence, batch_size):
    random.shuffle(sequence)
    return [sequence[i:i + batch_size]
            for i in range(0, len(sequence), batch_size)]


class NeuralNetwork:
    def __init__(self, layers):
        self.layers = layers
        self.biases = [np.random.randn(i, 1) for i in layers[1:]]
        self.weights = [np.random.randn(i, j)
                        for j, i in zip(layers[:-1], layers[1:])]

    def train_with_sgd(self, training_data, epochs=30, batch_size=10,
                       learning_rate=3.0):

        for epoch in range(epochs):
            batches = randomly_batch(training_data, batch_size)
            for batch in batches:
                self.update_weights_and_biases(batch, learning_rate)

            print(f"Epoch {epoch+1} complete")

    def update_weights_and_biases(self, example_data, learning_rate):
        step_size = learning_rate / len(example_data)
        dCdb, dCdw = self.backpropagate(example_data)
        self.descend_gradient(step_size, dCdb, dCdw)

    def descend_gradient(self, step_size, dCdb, dCdw):
        self.biases = [b - step_size * db
                       for b, db in zip(self.biases, dCdb)]
        self.weights = [w - step_size * dw
                        for w, dw in zip(self.weights, dCdw)]

    def backpropagate(self, training_data):
        dCdb = [np.zeros(b.shape) for b in self.biases]
        dCdw = [np.zeros(w.shape) for w in self.weights]

        for (x, y) in training_data:
            zs, activations = self.feedforward(x)

            delta = [np.zeros(b.shape) for b in self.biases]  # init
            delta[-1] = self.output_layer_error(zs, activations, y)

            for lyr in range(2, len(self.layers)):
                delta[-lyr] = self.hidden_layer_error(lyr, delta, zs)

            # FIXME (dCdb/w update - overwrites, throwing away value for all but last example)
            for lyr in range(len(self.layers)):
                dCdb[-lyr] = delta[-lyr]
                dCdw[-lyr] = np.dot(delta[-lyr], activations[-lyr-1].transpose())

        return dCdb, dCdw

    @staticmethod
    def output_layer_error(zs, activations, y):
        return (quadratic_cost_deriv(activations[-1], y)
                * sigmoid_deriv(zs[-1]))

    def hidden_layer_error(self, layer, delta, zs):
        return (np.matmul(np.transpose(self.weights[-layer+1]),
                          delta[-layer+1])
                * sigmoid_deriv(zs[-layer]))

    def feedforward(self, x):
        zs, activations = [], [x]
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, x) + b
            x = sigmoid(z)  # input for next layer
            zs.append(z), activations.append(x)

        return zs, activations

    def evaluate(self, test_data):
        num_correct = sum(self.classify(x) == y for x, y in test_data)
        print(f"{num_correct} / {len(test_data)} correct")

    def classify(self, x):
        _, activations = self.feedforward(x)
        return np.argmax(activations[-1])


if __name__ == "__main__":
    start = timeit.default_timer()

    training_data_set, _, test_data_set = load_mnist_data_set()

    net = NeuralNetwork([784, 30, 10])
    net.train_with_sgd(training_data_set)
    net.evaluate(test_data_set)

    stop = timeit.default_timer()
    print(f"Runtime: {stop - start} seconds")
