import numpy as np

def softmax(x):
    exp_values = np.exp(x - np.max(x, axis=0, keepdims=True))
    probabilities = exp_values / np.sum(exp_values, axis=0, keepdims=True)
    
    return probabilities

class ReLU:
    @staticmethod
    def activate(z):
        return np.maximum(0, z)
    
    @staticmethod
    def derivative(z):
        return np.where(z > 0, np.ones((len(z), len(z.T))), np.zeros((len(z), len(z.T))))

class Sigmoid:
    @staticmethod
    def activate(z):
        return 1/(1 + np.exp(-z))
    
    @classmethod
    def derivative(cls, z):
        return cls.activate(z) * (1 - cls.activate(z))
