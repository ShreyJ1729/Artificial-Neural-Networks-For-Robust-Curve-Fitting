import random
import numpy as np
import matplotlib.pyplot as plt
import math
import time
start_time=time.time()

class Network:

    def __init__(self, sizes):
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]

    def feedforward(self, a):
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a)+b)
        return a

    def SGD(self, training_data, epochs, mini_batch_size, eta):
        n = len(training_data)
        self.progress_epochs=[]
        for j in range(epochs):
            random.shuffle(training_data)
            mini_batches = [
                training_data[k:k+mini_batch_size]
                for k in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)
            cost_epoch=self.MSE(training_data_)
            self.progress_epochs.append(cost_epoch)

    def update_mini_batch(self, mini_batch, eta):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.weights = [w-(eta/len(mini_batch))*nw
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b-(eta/len(mini_batch))*nb
                       for b, nb in zip(self.biases, nabla_b)]

    def backprop(self, x, y):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # feedforward
        activation = x
        activations = [x] # list to store all the activations, layer by layer
        zs = [] # list to store all the z vectors, layer by layer
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation)+b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        # backward pass
        delta = self.cost_derivative(activations[-1], y) * \
            sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        # Note that the variable l in the loop below is used a little
        # differently to the notation in Chapter 2 of the book.  Here,
        # l = 1 means the last layer of neurons, l = 2 is the
        # second-last layer, and so on.  It's a renumbering of the
        # scheme in the book, used here to take advantage of the fact
        # that Python can use negative indices in lists.
        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1])
        return (nabla_b, nabla_w)

    def cost_derivative(self, output_activations, y):
        return (output_activations-y)
    def MSE(self, data):
        cost = [(i[1]-float(self.feedforward(i[0])*maximum))**2 for i in data]
        return sum(cost)
def sigmoid(z):
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
    return sigmoid(z)*(1-sigmoid(z))

def g(x):
    return np.sin(np.pi*x/5) + np.sqrt(x)

def f(x):
    return g(x) + np.random.normal(g(x), g(x)/10)


num_epochs = 300
learning_rate = 1
training_bounds = [1, 15]
num_training_samples = 40
poly_degree = 5

training_data = [(i, f(i)) for i in np.linspace(training_bounds[0], training_bounds[1], num_training_samples)]
training_data_ = training_data
training_data_x = [i[0] for i in training_data]
training_data_y = [i[1] for i in training_data]

maximum=max([i[1] for i in training_data])
training_data_scaled = [(i[0], i[1]/maximum) for i in training_data]

net=Network([1, 5, 1])
net.SGD(training_data_scaled, num_epochs, 1, learning_rate)

poly = np.poly1d(np.polyfit(training_data_x, training_data_y, poly_degree))

x=np.linspace(training_bounds[0], training_bounds[1], 100)

actual_y=f(x)
predicted_y=[float(net.feedforward(i)*maximum) for i in x]
plt.plot(x, predicted_y)
plt.plot(x, poly(x))
plt.plot(x, actual_y)
print(net.MSE(training_data))
plt.show()
epochs=[i for i in range(1, num_epochs+1)]
print(net.progress_epochs)
plt.plot(epochs, net.progress_epochs)
plt.show()
input()
