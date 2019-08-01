from sklearn.neural_network import MLPRegressor
import numpy as np
import matplotlib.pyplot as plt
import random
import time

squares = [];

k=random.randint(1, 5);
l=random.randint(1, 5);
m = (random.randint(1, 6))/3;
p = (random.randint(300, 400)/100);
q = (random.randint(100, 200)/100);
functions = ["((k/l)*(x/l)**(k-1) * np.exp(-(x/l)**(k)))", "1/x", "-(x-p)**2+q*x*np.cos(x)+5", "(np.sin(x))**2+np.sqrt(x)", "k**x" , "k*m*x/l", "m*x*np.sin(3*m*x-2)", "(x**k+1)/(k*x)", "np.sqrt(x**2+m)-np.sqrt(k*2*x)", "-m*x**4+k*x**3"];
number_neurons = int(input("Number of neurons in hidden layer of ANN: "));


random_selection = input("Select function at random? (y/n)");

if (random_selection=="y"):
    functionindex = random.randint(1, 10)
else:
    functionindex = int(input("Select function class. Enter integer from 1 to 10: "))-1;


def MSE(predicted_data_y, actual_data_y):
    print(len(predicted_data_y)==len(actual_data_y))
    for i in range(0, len(predicted_data_y)):
        squares.append((predicted_data_y[i] - actual_data_y[i])**2)
    return sum(squares)/len(predicted_data_y)

def f(x):
    return eval(functions[functionindex])

def g(x):
    return (f(x) + np.random.normal(f(x), abs(f(x)/5)))/2

all_data_x = np.arange(0.01, 6, 0.3)
all_data = [(i, g(i)) for i in all_data_x]

polydeg = int(input("Poly degree: "))

total_start = time.time()

training_data = all_data[2:-2]
test_data = []
test_data.extend(tuple(all_data[0:2]))
test_data.extend(tuple(all_data[-2:]))
graph_min = min([i[0] for i in test_data])
graph_max = max([i[0] for i in test_data])

x = np.array([i[0] for i in training_data]).reshape(-1, 1)
y = np.array([i[1] for i in training_data]).ravel()
poly_x = [i[0] for i in training_data]

ann_start = time.time()
nn = MLPRegressor(hidden_layer_sizes=(number_neurons),
                  activation='tanh', solver='lbfgs')

nn.fit(x, y)
ann_end = time.time()

test_data_x = [i[0] for i in test_data]
test_data_y = [i[1] for i in test_data]

all_data_x = [i[0] for i in all_data]
all_data_y = [i[1] for i in all_data]

test_graph_x = np.arange(0.8 * graph_min, 1.2*graph_max, 0.01).reshape(-1, 1)
ann_y = nn.predict(test_graph_x)

fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.scatter(x, y, s=5, c='b', marker="o", label='Training Data')
ax1.scatter(test_data_x, test_data_y, s=5, c='black', marker="o", label='Non-training Data')
ax1.plot(test_graph_x, ann_y, c='r', label='ANN Prediction')
all_data_x = np.arange(0.01, 7.2, 0.1)
all_data = [(i, f(i)) for i in all_data_x]
all_data_y = [i[1] for i in all_data]
ax1.plot(all_data_x, all_data_y, c='g', label='True Function')

poly_start=time.time()
poly = np.poly1d(np.polyfit(poly_x, y, polydeg))
poly_y=[poly(i) for i in test_graph_x]
ax1.plot(test_graph_x, poly_y, c='y', label = 'Poly Prediction')
poly_end=time.time()

ann_all_y = [nn.predict(np.array(i).reshape(-1, 1)) for i in all_data_x]
poly_all_y = [poly(i) for i in all_data_x]

plt.legend()

total_end=time.time()

print("Function Class Chosen: ", functions[functionindex])
print("k = ", k)
print("l = ", l)
print("m = ", m)
print("p = ", p)
print("q = ", q)
print("ANN MODEL:")
print("Mean-Squared-Error: ", sum(MSE(ann_all_y, all_data_y)))
print("Time Elapsed: ", 1000*(ann_end - ann_start), "milliseconds")
print("")
print("____________________________________")
print("POLYNOMIAL MODEL:")
print("Time Elapsed: ", (poly_end - poly_start))
print("Mean-Squared-Error: ", sum(MSE(poly_all_y, all_data_y)))
print("")
print("Total Time Elapsed: ", 1000*(total_end - total_start), " milliseconds")
plt.show()
