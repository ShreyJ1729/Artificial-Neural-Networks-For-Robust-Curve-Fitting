import random
import numpy as np
import matplotlib.pyplot as plt
import time
start_time=time.time()

class Network:

    def __init__(self, layers):
        self.num_layers = len(layers)
        self.layers = layers
        self.biases = [np.random.randn(y, 1) for y in layers[1:]]
        self.weights = [np.random.randn(y, x) for x, y in zip(layers[:-1], layers[1:])]

    def forwardpropagate(self, a):
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a)+b)
        return a

    def SGD(self, training_data, epochs, mini_batch_size, eta,
            test_data=None):
        if test_data: n_test = len(test_data)
        n = len(training_data)
        self.progress_epochs=[]
        for j in range(epochs):
            random.shuffle(training_data)
            mini_batches = [
                training_data[k:k+mini_batch_size]
                for k in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)
            cost_epoch=self.MSE(training_data)
            self.progress_epochs.append(cost_epoch)

    def update_mini_batch(self, mini_batch, eta):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backpropagate(x, y)
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.weights = [w-(eta/len(mini_batch))*nw
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b-(eta/len(mini_batch))*nb
                       for b, nb in zip(self.biases, nabla_b)]

    def backpropagate(self, x, y):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # forwardpropagate
        activation = x
        activations = [x] # list to store all the activations, layer by layer
        weighted_sums = [] # list to store all the z vectors, layer by layer
        for b, w in zip(self.biases, self.weights):
            weighted_sum = np.dot(w, activation)+b
            weighted_sums.append(weighted_sum)
            activation = sigmoid(weighted_sum)
            activations.append(activation)
        # backward pass
        delta = self.MSE_derivative(activations[-1], y) * \
            sigmoid_prime(weighted_sums[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        # Note that the variable l in the loop below is used a little
        # differently to the notation in Chapter 2 of the book.  Here,
        # l = 1 means the last layer of neurons, l = 2 is the
        # second-last layer, and so on.  It's a renumbering of the
        # scheme in the book, used here to take advantage of the fact
        # that Python can use negative indices in lists.
        for l in range(2, self.num_layers):
            weighted_sum = weighted_sums[-l]
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sigmoid_prime(weighted_sum)
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1])
        return (nabla_b, nabla_w)

    def MSE_derivative(self, output_activations, y):
        return (output_activations-y)
    def MSE(self, training_data):
        training_x=[i[0] for i in training_data]
        training_y=[i[1] for i in training_data]
        cost=[((training_y[a]-self.forwardpropagate(b))**2)/2 for a, b in zip(range(0, len(training_x)), training_x)]
        MSE_cost=sum(cost)/len(cost)
        return float(MSE_cost)
def sigmoid(z):
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
    return sigmoid(z)*(1-sigmoid(z))

def f(x):
    return np.sin(np.pi*x/5) + np.sqrt(x)

training_lower_bound=float(input("Enter Training Data Lower Bound: "))
training_upper_bound=float(input("Enter Training Data Upper Bound: "))
num_training_samples=int(input("Enter Number of Training Samples: "))

training_data=[(i, f(i)+np.random.normal(f(i), f(i)/20)) for i in np.linspace(training_lower_bound, training_upper_bound, num_training_samples)]
maximum = max([i[1] for i in training_data])
training_data_scaled=[]

for i in range(0, len(training_data)):
    training_data_scaled.append((training_data[i][0], training_data[i][1]/maximum))

print(maximum)
num_epochs = int(input("Enter Number of Epochs to Train: "))
learning_rate = float(input("Enter Learning Rate: "))
mini_batch_size=int(input("Enter Mini-batch Size: "))
net1=Network([1, 5, 1])

net1.SGD(training_data_scaled, num_epochs, mini_batch_size, learning_rate)

x=np.linspace(training_lower_bound, training_upper_bound, num_training_samples)
actual_y=f(x)

plt.plot(x, actual_y)
predicted_y=[float(net1.forwardpropagate(i)) for i in x]

plt.plot(x, predicted_y)
poly_array=np.polyfit(x, actual_y, 100)
poly=np.poly1d(poly_array)
plt.plot(x, poly(x))
cost=0
for i in x:
    cost += (actual_y-predicted_y)**2
print("Artificial Neural Network Model:")
print("Mean-Squared Error Cost: ", sum(cost)/(2*len(cost)))
print("Average percentage accuracy: ", 100-sum(abs((100-100*(predicted_y/actual_y))))/len(predicted_y), "%")
print("--- %s seconds elapsed ---" % (time.time() - start_time))
plt.show()
input()
x=[i for i in range(1, num_epochs+1)]
print(x)
print(net1.progress_epochs)
input()
plt.plot(x, net1.progress_epochs)
plt.show()
input()
