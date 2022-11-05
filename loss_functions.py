import numpy as np

from activation_functions import softmax

def categorical_cross_entropy(y_pred, y_true):
    samples = len(y_pred)
    y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)

    # when there is only one instance
    if len(y_true[y_true==1]) == 1:
        correct_confidences = np.sum(y_pred_clipped*y_true)

    # when there are multiple instances

    # without one hot encoding e.g. [1, 0, 2]
    elif y_true.ndim == 1:
        correct_confidences = y_pred_clipped[samples, y_true]

    # with one hot encoding e.g [[1, 0, 0]
    #                            [0, 1, 0]
    #                            [0, 0, 1]]

    elif y_true.ndim == 2:
        correct_confidences = np.sum(y_pred_clipped*y_true, axis=0)

    negative_log_likelihood = -np.log(correct_confidences)

    return negative_log_likelihood

class CategoricalCrossEntropyWithSoftmax:
    @staticmethod
    def activate(z):
        return softmax(z)

    @staticmethod
    def loss(probabilities, y):
        return categorical_cross_entropy(probabilities, y)
    
    @staticmethod
    def derivative(a, y):
        return a - y

class MSE:
    @staticmethod
    def loss(output, y):
        return np.mean((y - output)**2, axis=1)
    
    @staticmethod
    def derivative(output, y):
        2 * np.mean(y - output, axis=1)
