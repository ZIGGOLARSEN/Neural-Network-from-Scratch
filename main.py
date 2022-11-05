import pandas as pd
from sklearn.preprocessing import StandardScaler

from NeuralNetwork import NeuralNetwork
from Layer import Layer
from loss_functions import *
from activation_functions import *


scaler = StandardScaler()

train_data = pd.read_csv('mnist_train.csv')
test_data = pd.read_csv('mnist_test.csv')

train_labels = pd.get_dummies(train_data.label).to_numpy().T

test_labels = pd.get_dummies(test_data.label).to_numpy().T

train_data = np.array(train_data.drop('label', axis=1)).T
test_data = np.array(test_data.drop('label', axis=1)).T

train_data = scaler.fit_transform(train_data)
test_data = scaler.fit_transform(test_data)



batch_size = 100
neural_net = NeuralNetwork([
    Layer(784, 100, batch_size, Sigmoid),
    Layer(100, 10, batch_size, CategoricalCrossEntropyWithSoftmax)
], 
train_data, train_labels, 0.5, batch_size=batch_size)

neural_net.learn(15)
print(neural_net.predict(test_data, test_labels)[1])
