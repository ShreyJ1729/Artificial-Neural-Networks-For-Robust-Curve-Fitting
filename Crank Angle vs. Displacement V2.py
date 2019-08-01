from sklearn.neural_network import MLPRegressor
import numpy as np
import matplotlib.pyplot as plt
import time

polydeg = int(input("Poly degree? "))

total_start = time.time()

training_data = [(121, 56), (136, 58.3), (151, 60), (166, 61.3), (181, 62.2), (196, 63.2), (211, 64.3), (226, 65.6), (241, 67), (256, 68), (271, 68.4), (286, 68.1), (301, 66.8), (316, 64.5)]

x = np.array([i[0]/100 for i in training_data]).reshape(-1, 1)
y = np.array([i[1] for i in training_data]).ravel()
poly_x = [i[0]/100 for i in training_data]

ann_start = time.time()
nn = MLPRegressor(hidden_layer_sizes=(8, 8),
                  activation='tanh', solver='lbfgs', max_iter = 1000)

nn.fit(x, y)
ann_end = time.time()


test_data = [(91, 50.9), (106, 53.5), (331, 61.2), (346, 57.4)]
test_data_x = [i[0]/100 for i in test_data]
test_data_y = [i[1] for i in test_data]
test_graph_x = np.arange(0.7, 3.5, 0.01).reshape(-1, 1)
test_ann_y = nn.predict(test_graph_x)

fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.scatter(x, y, s=5, c='b', marker="o", label='Training Data')
ax1.scatter(test_data_x, test_data_y, s=5, c='g', marker="o", label='Test Data')
ax1.plot(test_graph_x,test_ann_y, c='r', label='ANN Prediction')


poly_start=time.time()
poly = np.poly1d(np.polyfit(poly_x, y, polydeg))
test_poly_y=[poly(i) for i in test_graph_x]
ax1.plot(test_graph_x, test_poly_y, c='y', label='Poly Prediction (Degree 5)')
poly_end=time.time()

plt.legend()
total_end=time.time()
print("ANN MODEL:")
print("Time Elapsed: ", 1000*(ann_end - ann_start), "milliseconds")
print("____________________________________")
print("POLYNOMIAL MODEL:")
print("Time Elapsed: ", (poly_end - poly_start))
print("Total Time Elapsed: ", 1000*(total_end - total_start), " milliseconds")
plt.show()
