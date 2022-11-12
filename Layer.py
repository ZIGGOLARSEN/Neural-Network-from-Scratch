import numpy as np

from activation_functions import *

class Layer:
    def __init__(self, num_neurons_this_layer, num_neurons_next_layer, input_size, activation_function, reg_type, lmbda=0):
        self.num_neurons_this_layer = num_neurons_this_layer
        self.num_neurons_next_layer = num_neurons_next_layer
        self.input_size = input_size
        self.activation_function = activation_function
        self.reg_type = reg_type
        self.lmbda = lmbda
        self.initialize_weights_and_biases()

    def __repr__(self):
        return f'Layer({self.num_neurons_this_layer}) --> {self.num_neurons_next_layer}'

    def initialize_weights_and_biases(self):
        # we initialize weights to be random matrix shape of which is 
        # defined according to the layer structure of future neural network
        # biases are initialized to be zero

        n = self.num_neurons_this_layer
        k = self.num_neurons_next_layer

        self.W = np.random.randn(k, n) / np.sqrt(n)
        self.b = np.zeros((k, self.input_size))

    def forward(self, input):
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

    def error(self, delta_l):
        derivative_of_activation = self.activation_function.derivative(self.z)
        delta_l = delta_l * derivative_of_activation

        return delta_l        

    def backward(self, label_or_error, learn_rate):
        # output error and hidden layer errors are different, thats why we have parameter 
        # label_or_error: if the layer is output layer label_or_error is label
        # otherwise its error returned from previous layer

        delta_l = self.error(label_or_error)

        # derivatives for gradient
        # lmbda parameter controls l1 or l2 regularization 
        # if set to zero network is unregularized

        if self.reg_type == 'l1':
            w_sgn = np.sign(self.W)

            dweights = delta_l@self.a.T + self.lmbda*w_sgn
            self.W -= learn_rate/len(self.z.T)*dweights - learn_rate * self.lmbda * w_sgn

        elif self.lmbda != 0:
            dweights = delta_l@self.a.T + self.lmbda*self.W
            self.W -= learn_rate/len(self.z.T)*dweights - learn_rate * self.lmbda * self.W

        else:
            dweights = delta_l@self.a.T
            self.W -= learn_rate/len(self.z.T)*dweights

        dbiases = delta_l
        self.b -= learn_rate/len(self.z.T)*dbiases
        
        # returning self's error to do same operations on previous
        # selfs as we did here
        return self.W.T@delta_l


class Output(Layer):
    def __init__(self, num_neurons_this_layer, num_neurons_next_layer, input_size, activation_function, reg_type, lmbda):
        super().__init__(num_neurons_this_layer, num_neurons_next_layer, input_size, activation_function, reg_type, lmbda)

    def error(self, label):        
        delta_l = self.activation_function.derivative(self.output, label)
        return delta_l
