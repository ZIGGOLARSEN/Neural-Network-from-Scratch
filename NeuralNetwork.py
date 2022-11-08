import numpy as np

from loss_functions import *
from activation_functions import *
from Layer import Layer

class NeuralNetwork:
    def __init__(self, network: list[Layer], data, y, learn_rate, batch_size, reg_type = 'l2', lmbda=0, loss = CategoricalCrossEntropyWithSoftmax.loss):
        self.network = network
        self.data = data
        self.y = y
        self.batch_size = batch_size
        self.learn_rate = learn_rate
        self.reg_type = reg_type
        self.lmbda = lmbda
        self.loss = loss

        self.batches, self.labels = self.divide_data(self.data, self.y)


    def divide_data(self, data, y):
        # case where data has less instances than given batch size for network
        if len(self.data.T) <= self.batch_size:
            batches = [data]
            labels = [y]
            return

        # cases where data is several instances in one matrix (it has more that one rows)
        try:
            # we divide data into batch_size sized batches
            # first we try to divide it without remainder 
            # but if it raises an exception we handle it in the ValueError exception
            
            self.batch_count = len(data.T) // self.batch_size
            if len(data.T) % self.batch_size != 0: raise ValueError
            batches = np.split(data, self.batch_count, axis=1)
            labels = np.split(y, self.batch_count, axis=1)
        
        except ValueError:
            # here we divide data into as many batches as we can
            # and add individual bathces to the batches list
 
            last_idx = self.batch_count * self.batch_size
            batches = np.split(data[:, :last_idx], self.batch_count, axis=1)
            labels = np.split(y[:, :last_idx], self.batch_count, axis=1)

            # after we reach the point where it isn't able to divide anymore we
            # append remaining instances into last batch regardless their size
            batches.append(data[:, last_idx:])
            labels.append(y[:, last_idx:])

        return batches, labels

    def feed_forward(self, batch):
        # we feed all instances into one particular batch to the network
        # and then we get the corresponding output

        input = batch

        # feeding one batch to the network
        for layer in self.network:
            output = layer.calculate_output(input)
            input = output

        return output

    def cost(self):
        # we iterate through zip of networks outputs and target labels
        # and pass them into the loss function one batch at a time. 
        # then we compute average cost of each batch

        cost = np.mean([self.loss(output, label) for output, label in zip(self.outputs, self.labels)])

        return cost

    def update_gradients(self, layer: Layer, layer_type, label=None, delta_L=None):
        # output error and hidden layer errors are different 
        # thats why we need to handle them separately
        if layer_type == 'output':
            delta_l = layer.activation_function.derivative(layer.output, label)
        else:
            derivative_of_activation = layer.activation_function.derivative(layer.z)
            delta_l = delta_L * derivative_of_activation

        # derivatives for gradient
        # self.lmbda parameter controls l1 or l2 regularization 
        # if set to zero network is unregularized

        if self.reg_type == 'l1':
            w_sgn = np.sign(layer.W)

            dweights = delta_l@layer.a.T + self.lmbda*w_sgn/len(layer.z.T)
            layer.W -= self.learn_rate/len(layer.z.T) * dweights - self.learn_rate * self.lmbda * w_sgn / len(self.data.T)
        else:
            dweights = delta_l@layer.a.T + self.lmbda*layer.W/len(layer.z.T)
            layer.W -= self.learn_rate/len(layer.z.T) * dweights - self.learn_rate * self.lmbda * layer.W / len(self.data.T)

        dbiases = delta_l
        layer.b -= self.learn_rate/len(layer.z.T) * dbiases

        # returning layer's error to do same operations on previous
        # layers as we did here
        return layer.W.T@delta_l

    def backprop(self, label):
        output_error = self.update_gradients(self.network[-1], 'output', label)

        # iterating through hidden layers and updating gradients
        for layer in self.network[-2::-1]:
            output_error = self.update_gradients(layer, 'hidden', delta_L = output_error)

    def learn(self, epochs):
        # for each batch in the data we feed forwrd the batch into the network
        # then propagate backwards and update weights and biases along the way
        # we do this for each epoch

        for epoch in range(epochs):
            batch_accuracies = []
            for batch, label in zip(self.batches, self.labels):
                output = self.feed_forward(batch)
                
                # we calculate batch accuracy and append it to array of batch accuracies 
                # to use them later for determinig accuracy on each epoch 
                prediction = np.argmax(output, axis=0)
                accuracy = self.accuracy(prediction, label)
                batch_accuracies.append(accuracy)

                self.backprop(label)

            # printing training data accuracies
            print(f'Epoch {epoch+1}: accuracy -> {np.mean(batch_accuracies):.6f}')
            batch_accuracies = []

    def predict(self, test_data, test_labels):
        predictions = []
        accuracies = []
        
        # we split test data and test lables into batches
        # then we feed forwrd each batch to the trained network
        # and get predictions and calculate accuracies of those predictions

        for batch, label in zip(*self.divide_data(test_data, test_labels)):
            output = self.feed_forward(batch)
            
            # networks prediction is the index a which
            # there is a highest output probability 
            prediction = np.argmax(output, axis=0)
            predictions.append(prediction)

            accuracies.append(self.accuracy(prediction, label))
        
        return predictions, np.round(np.mean(accuracies), 6)

    def accuracy(self, prediction, label):
        accuracy = np.sum(label[prediction, range(len(label.T))] ) / len(label.T)
        return accuracy
