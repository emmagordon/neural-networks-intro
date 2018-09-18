import gzip
import pickle

import numpy as np


def load_mnist_data_set():
    """Loads the MNIST data set - this function originates from
    https://github.com/mnielsen/neural-networks-and-deep-learning,
    save slight modification. Copyright Michael Nielsen.
    """

    def vectorise(x):
        # Aiming for output activation of 1 for correct label,
        # and 0 for everything else.
        v = np.zeros((10, 1))
        v[x] = 1.0
        return v

    with gzip.open('data/mnist.pkl.gz', 'rb') as f:
        tr_d, va_d, te_d = pickle.load(f, encoding='latin1')

    training_inputs = [np.reshape(x, (784, 1)) for x in tr_d[0]]
    training_results = [vectorise(y) for y in tr_d[1]]
    training_data = list(zip(training_inputs, training_results))

    validation_inputs = [np.reshape(x, (784, 1)) for x in va_d[0]]
    validation_data = list(zip(validation_inputs, va_d[1]))

    test_inputs = [np.reshape(x, (784, 1)) for x in te_d[0]]
    test_data = list(zip(test_inputs, te_d[1]))

    return training_data, validation_data, test_data
