import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
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

    def SGD(self, training_data, mini_batch_size, eta):
        n = len(training_data)
        self.progress_epochs=[]
        epoch=0
        predicted_y=[float(self.feedforward(i)*maximum) for i in x]
        while percent_accuracy(predicted_y, actual_y) < 94:
            cost_epoch=self.MSE(training_data_)
            self.progress_epochs.append(cost_epoch)
            if epoch % 100 == 0:
                print("Cost For Epoch ", epoch, ": ", cost_epoch)
                print("Average Accuracy ", percent_accuracy(predicted_y, actual_y))
            random.shuffle(training_data)
            mini_batches = [
                training_data[k:k+mini_batch_size]
                for k in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)
            epoch += 1
            predicted_y=[float(net.feedforward(i)*maximum) for i in x]
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
        activation = x
        activations = [x]
        zs = []
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation)+b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        delta = self.cost_derivative(activations[-1], y) * \
            sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
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
        return sum(cost)/len(cost)
def sigmoid(z):
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
    return sigmoid(z)*(1-sigmoid(z))

def g(x):
    return -0.7/(np.sin(x/2-6))+3

def f(x):
    return (g(x) + np.random.normal(g(x), abs(g(x)/10)))/2

def percent_accuracy(predicted_y_data, actual_y_data):
    percent_accuracy_all=[]
    for i, j in zip(predicted_y_data, actual_y_data):
        if i<j:
            accuracy=100*abs(j-i)/j
        else:
            accuracy=100*abs(i-j)/i
        if accuracy > 100:
            accuracy=accuracy-100
        percent_accuracy_all.append(abs(accuracy))
    average_accuracy=100-sum(percent_accuracy_all)/len(percent_accuracy_all)
    return average_accuracy

def MSE_poly(test_data):
    cost=[(i[1]-poly(i[0]))**2 for i in test_data]
    return sum(cost)/len(cost)
learning_rate = 1
training_bounds = [0, 5.2]
num_training_samples = 20
poly_degree = 5

training_data = [(i, f(i)) for i in np.linspace(training_bounds[0], training_bounds[1], num_training_samples)]
training_data_ = training_data
training_data_x = [i[0] for i in training_data]
training_data_y = [i[1] for i in training_data]

maximum=max([i[1] for i in training_data])
training_data_scaled = [(i[0], i[1]/maximum) for i in training_data]

x=np.linspace(training_bounds[0], training_bounds[1], 100)
general_trend_y=[g(i) for i in x]
actual_y=f(x)
net=Network([1, 5, 5, 1])
net.SGD(training_data_scaled, 1, learning_rate)
predicted_y=[float(net.feedforward(i)*maximum) for i in x]



general_trend_tuples=[(i, g(i)) for i in np.linspace(training_bounds[0], training_bounds[1], num_training_samples)]

end_time=time.time()

poly_start=time.time()
poly = np.poly1d(np.polyfit(training_data_x, training_data_y, poly_degree))
poly_predicted_y=[poly(i) for i in x]
plt.plot(x, poly_predicted_y)
poly_end=time.time()

plt.plot(x, predicted_y)
plt.plot(x, actual_y, label='Function with Noise')
plt.plot(x, general_trend_y, label='Pure Function')
print("ARTIFICIAL NEURAL NETWORK MODEL:")
print("Mean-Squared-Error Cost: ", net.MSE(general_trend_tuples))
print("Average Percentage Accuracy: ", percent_accuracy(predicted_y, actual_y), "%")
print(" --- %s Seconds Elapsed ---" % (end_time - start_time))
print("POLYNOMIAL REGRESSION MODEL:")
print("Mean-SquareError Cost: ", MSE_poly(general_trend_tuples))
print("Average Percentage Accuracy: ", percent_accuracy(poly_predicted_y, actual_y), "%")
print(" --- %s Seconds Elapsed ---" % (poly_end - poly_start))

print("Red - Pure Function")
print("Green - Noisy Function")
print("Blue - Polynomial")
print("Orange - ANN model")
print()
plt.show()
epochs=[i for i in range(1, len(net.progress_epochs)+1)]
plt.plot(epochs, net.progress_epochs)
print("Epoch VS. Cost Graph Shown")
plt.show()
input()
