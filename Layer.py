import numpy as np

from activation_functions import *

class Layer:
    def __init__(self, num_neurons_this_layer, num_neurons_next_layer, input_size, activation_function = ReLU):
        self.num_neurons_this_layer = num_neurons_this_layer
        self.num_neurons_next_layer = num_neurons_next_layer
        self.input_size = input_size
        self.activation_function = activation_function
        self.initialize_weights_and_biases()

    def __repr__(self):
        return f'Layer({self.num_neurons_this_layer}) --> {self.num_neurons_next_layer}'

    def initialize_weights_and_biases(self):
        # we initialize weights to be random matrix shape of which is 
        # defined according to the layer structure of future neural network
        # biases are initialized to be zero

        n = self.num_neurons_this_layer
        k = self.num_neurons_next_layer

        self.W = np.random.rand(k, n)
        self.b = np.zeros((k, self.input_size))

    def calculate_output(self, input):
        # we simply calculate weighted sum of inputs from previous layer plus bias
        # and pass it through the activation function        

        try:
            self.a = input
            self.z = self.W@self.a + self.b
            self.output = self.activation_function.activate(self.z)        

        except ValueError:
            self.b = np.zeros((self.num_neurons_next_layer, len(self.a.T)))

        finally:
            self.z = self.W@self.a + self.b
            self.output = self.activation_function.activate(self.z)

        return self.output
