from sklearn.neural_network import MLPRegressor
import numpy as np
import matplotlib.pyplot as plt
import time

total_start = time.time()

training_data = [(91, 50.9), (106, 53.5), (121, 56), (136, 58.3), (151, 60), (166, 61.3), (181, 62.2), (196, 63.2), (211, 64.3), (226, 65.6), (241, 67), (256, 68), (271, 68.4), (286, 68.1), (301, 66.8), (316, 64.5), (331, 61.2), (346, 57.4)]

x = np.array([i[0]/100 for i in training_data]).reshape(-1, 1)
y = np.array([i[1] for i in training_data]).ravel()



nn = MLPRegressor(hidden_layer_sizes=(5),
                  activation='tanh', solver='lbfgs')

nn.fit(x, y)
test_x = np.arange(0, 4 , 0.01).reshape(-1, 1)
test_y = nn.predict(test_x)
fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.scatter(x, y, s=5, c='b', marker="o", label='Real Data')
ax1.plot(test_x,test_y, c='r', label='ANN Prediction')

plt.legend()
total_end=time.time()

print("Total Time Elapsed: ", 1000*(total_end - total_start), " milliseconds")
plt.show()
