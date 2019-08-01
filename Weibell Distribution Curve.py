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

polydeg = int(input("Poly degree?"))

total_start = time.time()

all_data = [(1, 0.0522), (1.08, 0.1595), (1.15, 0.1595), (1.46, 0.4213), (1.55, 0.536), (2.05, 0.4968), (2.2, 0.4767), (2.6, 0.3844), (3.1, 0.2487), (3.5, 0.1569), (3.84, 0.0991), (4, 0.0782), (4.4, 0.041), (4.68, 0.025), (5, 0.0136)]
training_data = [(1.15, 0.1595), (1.46, 0.4213), (1.55, 0.536), (2.05, 0.4968), (2.2, 0.4767), (2.6, 0.3844), (3.1, 0.2487), (3.5, 0.1569), (3.84, 0.0991), (4, 0.0782), (4.4, 0.041)]
test_data = [(1, 0.0522), (1.08, 0.1595), (4.68, 0.025), (5, 0.0136)]

graph_min = min([i[0] for i in test_data])
graph_max = max([i[0] for i in test_data])

x = np.array([i[0] for i in training_data]).reshape(-1, 1)
y = np.array([i[1] for i in training_data]).ravel()
poly_x = [i[0] for i in training_data]

ann_start = time.time()
nn = MLPRegressor(hidden_layer_sizes=(7),
                  activation='tanh', solver='lbfgs')

nn.fit(x, y)
ann_end = time.time()

test_data_x = [i[0] for i in test_data]
test_data_y = [i[1] for i in test_data]

all_data_x = [i[0] for i in all_data]
all_data_y = [i[1] for i in all_data]

test_graph_x = np.arange(0.9 * graph_min, 1.1*graph_max, 0.01).reshape(-1, 1)
ann_y = nn.predict(test_graph_x)

fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.scatter(x, y, s=5, c='b', marker="o", label='Training Data')
ax1.scatter(test_data_x, test_data_y, s=5, c='g', marker="o", label='Test Data')
ax1.plot(test_graph_x, ann_y, c='r', label='ANN Prediction')


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
print("Time Elapsed: ", (poly_end - poly_start))
print("Mean-Squared-Error: ", sum(MSE(poly_all_y, all_data_y)))
print("Total Time Elapsed: ", 1000*(total_end - total_start), " milliseconds")
plt.show()
