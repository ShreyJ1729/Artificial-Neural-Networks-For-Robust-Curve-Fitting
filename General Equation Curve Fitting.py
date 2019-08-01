from sklearn.neural_network import MLPRegressor
import numpy as np
import matplotlib.pyplot as plt
import time

squares = [];
def MSE(predicted_data_y, actual_data_y):
    print(len(predicted_data_y)==len(actual_data_y))
    for i in range(0, len(predicted_data_y)):
        squares.append((predicted_data_y[i] - actual_data_y[i])**2)
    return sum(squares)/len(predicted_data_y)

def f(x):
    return 5*np.sin(np.pi*x/3)

def g(x):
    return (f(x) + (f(x) + np.random.normal(f(x), abs(f(x)/10)))/2)

all_data_x = np.arange(0, 6, 0.1)
all_data = [(i, g(i)) for i in all_data_x]

polydeg = int(input("Poly degree?"))

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



test_data_x = [i[0] for i in test_data]
test_data_y = [i[1] for i in test_data]

all_data_x = [i[0] for i in all_data]
all_data_y = [i[1] for i in all_data]

test_graph_x = np.arange(0.8 * graph_min, 1.2*graph_max, 0.01).reshape(-1, 1)

fig = plt.figure()
ax1 = fig.add_subplot(111)


ann_start = time.time()
nn = MLPRegressor(hidden_layer_sizes=(15),
                  activation='tanh', solver='lbfgs')

nn.fit(x, y)
ann_y = nn.predict(test_graph_x)
ax1.plot(test_graph_x, ann_y, c='r', label='ANN Prediction')

ann_end = time.time()



ax1.scatter(x, y, s=5, c='b', marker="o", label='Training Data')
ax1.scatter(test_data_x, test_data_y, s=5, c='g', marker="o", label='Test Data')


poly_start=time.time()
poly = np.poly1d(np.polyfit(poly_x, y, polydeg))
poly_y=[poly(i) for i in test_graph_x]
ax1.plot(test_graph_x, poly_y, c='y', label='Poly Prediction')
poly_end=time.time()

ann_all_y = [nn.predict(np.array(i).reshape(-1, 1)) for i in all_data_x]
poly_all_y = [poly(i) for i in all_data_x]

plt.legend()
total_end=time.time()
print("ANN MODEL:")
print("Mean-Squared-Error: ", sum(MSE(ann_all_y, all_data_y)))
print("Time Elapsed: ", 1000*(ann_end - ann_start), "milliseconds")
print("____________________________________")
print("POLYNOMIAL MODEL:")
print("Time Elapsed: ", 1000*(poly_end - poly_start), "milliseconds")
print("Mean-Squared-Error: ", sum(MSE(poly_all_y, all_data_y)))
print("")
print("Total Time Elapsed: ", 1000*(total_end - total_start), " milliseconds")
plt.show()
