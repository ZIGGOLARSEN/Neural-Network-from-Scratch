import numpy as np

from loss_functions import *
from activation_functions import *
from Layer import Layer

class NeuralNetwork:
    def __init__(self, network: list[Layer], learn_rate, batch_size, loss = CategoricalCrossEntropyWithSoftmax.loss):
        self.network = network
        self.learn_rate = learn_rate
        self.batch_size = batch_size
        self.loss = loss


    def divide_data(self, data, y):
        # case where data has less instances than given batch size for network
        if len(data.T) <= self.batch_size:
            batches = [data]
            labels = [y]
            return

        # cases where data is several instances in one matrix (it has more that one rows)
        try:
            # we divide data into batch_size sized batches
            # first we try to divide it without remainder 
            # but if it raises an exception we handle it in the ValueError exception
            
            batch_count = len(data.T) // self.batch_size
            if len(data.T) % self.batch_size != 0: raise ValueError
            batches = np.split(data, batch_count, axis=1)
            labels = np.split(y, batch_count, axis=1)
        
        except ValueError:
            # here we divide data into as many batches as we can
            # and add individual bathces to the batches list
 
            last_idx = batch_count * self.batch_size
            batches = np.split(data[:, :last_idx], batch_count, axis=1)
            labels = np.split(y[:, :last_idx], batch_count, axis=1)

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
            output = layer.forward(input)
            input = output

        return output

    def backprop(self, label):
        # calculating output error to propagate it
        # backwards and update gradient
        output_error = self.network[-1].backward(label, self.learn_rate)

        # iterating through hidden layers and updating gradients
        for layer in self.network[-2::-1]:
            output_error = layer.backward(output_error, self.learn_rate)

    def learn(self, batches, labels, epochs):
        # for each batch in the data we feed forwrd the batch into the network
        # then propagate backwards and update weights and biases along the way
        # we do this for each epoch

        for epoch in range(epochs):
            batch_accuracies = []
            batch_costs = []

            for batch, label in zip(batches, labels):
                output = self.feed_forward(batch)
                batch_costs.append(self.batch_cost(output, label))

                # we calculate batch accuracy and append it to array of batch accuracies 
                # to use them later for determinig accuracy on each epoch 
                prediction = np.argmax(output, axis=0)
                accuracy = self.accuracy(prediction, label)
                batch_accuracies.append(accuracy)

                self.backprop(label)

            epoch_cost = self.cost(batch_costs)

            # printing training data costs and accuracies for each epoch
            print(f'Epoch {epoch+1}: loss -> {epoch_cost:.6f};  accuracy -> {np.mean(batch_accuracies):.6f}')

            batch_accuracies = []
            batch_costs = []


    def batch_cost(self, output, label):
        cost = np.mean(self.loss(output, label))
        return cost

    def cost(self, batch_costs):
        return np.mean(batch_costs)


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

    def fit(self, X, y, epochs, shuffle=False):
        if shuffle:
            indecies = np.random.permutation(np.arange(len(X.T)))
            X = X[:, indecies]
            y = y[:, indecies]

        batches, labels = self.divide_data(X, y)

        # if we fit the data first time, then self.batches and self.labels
        # do not exist yet so we create them
        # if we fit new data on top of previously fitted data, then 
        # self.batches and self.labels already exist so we concatinate 
        # them with new batches and labels
        
        if hasattr(self, 'batches') and hasattr(self, 'labels'):
            self.batches += batches
            self.labels += labels

        else:
            self.batches, self.labels = batches, labels

        # train network on new data for specified amount of epochs
        self.learn(batches, labels, epochs)
