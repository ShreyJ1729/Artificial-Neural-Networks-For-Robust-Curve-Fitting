from sklearn.neural_network import MLPRegressor
import numpy as np
import matplotlib.pyplot as plt
import random
import time

total_start = time.time()


def MSE(predicted_data_y, actual_data_y):
    squares = [];
    print(len(predicted_data_y)==len(actual_data_y))
    for i in range(0, len(predicted_data_y)):
        squares.append((predicted_data_y[i] - actual_data_y[i])**2)
    squares2 = squares
    squares = []
    return sum(squares2)/len(predicted_data_y)

text_file = open("ANN Output3.txt", "w")

MSEs = []
timeElapseds = []
activationFunctions = ["tanh", "relu", "logistic"]
minimizationMethods = ["sgd", "lbfgs"]
functions = ["((k/l)*(x/l)**(k-1) * np.exp(-(x/l)**(k)))", "1/x", "-(x-p)**2+q*x*np.cos(x)+5", "(np.sin(x))**2+np.sqrt(x)", "k**x" , "k*m*x/l", "m*x*np.sin(3*m*x-2)", "(x**k+1)/(k*x)", "np.sqrt(x**2+m)-np.sqrt(k*2*x)", "-m*x**4+k*x**3"];

MSEsAvgsNeurons = [];
timeElapsedAvgsNeurons = [];

MSEsAvgsSolvers = [];
timeElapsedAvgsSolvers = [];

MSEsAvgsAFunctions = [];
timeElapsedAvgsAFunctions = [];

MSEsAvgsFunctions = [];
timeElapsedAvgsFunctions = [];


for functionIndex in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]:
    for activationFunction in activationFunctions:
        for minimizationMethod in minimizationMethods:
            for numberNeurons in [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]:
                for w in range(0, 50):
                    k=random.randint(1, 5);
                    l=random.randint(1, 5);
                    m = (random.randint(1, 6))/3;
                    p = (random.randint(300, 400)/100);
                    q = (random.randint(100, 200)/100);
                    def f(x):
                        return eval(functions[functionIndex])

                    def g(x):
                        return (f(x) + np.random.normal(f(x), abs(f(x)/5)))/2

                    all_data_x = np.arange(0.01, 6, 0.3)
                    all_data = [(i, g(i)) for i in all_data_x]

                    training_data = all_data[2:-2]
                    test_data = []
                    test_data.extend(tuple(all_data[0:2]))
                    test_data.extend(tuple(all_data[-2:]))
                    graph_min = min([i[0] for i in test_data])
                    graph_max = max([i[0] for i in test_data])

                    x = np.array([i[0] for i in training_data]).reshape(-1, 1)
                    y = np.array([i[1] for i in training_data]).ravel()

                    ann_start = time.time()
                    nn = MLPRegressor(hidden_layer_sizes=(numberNeurons),
                                      activation=activationFunction,
                                      solver=minimizationMethod,
                                      max_iter=1000)

                    nn.fit(x, y)
                    ann_end = time.time()

                    test_data_x = [i[0] for i in test_data]
                    test_data_y = [i[1] for i in test_data]

                    all_data_x = [i[0] for i in all_data]
                    all_data_y = [i[1] for i in all_data]

                    test_graph_x = np.arange(0.8 * graph_min, 1.2*graph_max, 0.01).reshape(-1, 1)
                    ann_y = nn.predict(test_graph_x)

                    all_data_x = np.arange(0.01, 7.2, 0.1)
                    all_data = [(i, f(i)) for i in all_data_x]
                    all_data_y = [i[1] for i in all_data]

                    ann_all_y = [nn.predict(np.array(i).reshape(-1, 1)) for i in all_data_x]

                    MSEsAvgsNeurons.extend(MSE(ann_all_y, all_data_y))
                    timeElapsedAvgsNeurons.append(ann_end - ann_start)
                    
                    MSEsAvgsSolvers.extend(MSE(ann_all_y, all_data_y))
                    timeElapsedAvgsSolvers.append(ann_end - ann_start)
                    
                    MSEsAvgsAFunctions.extend(MSE(ann_all_y, all_data_y))
                    timeElapsedAvgsAFunctions.append(ann_end - ann_start)

                    MSEsAvgsFunctions.extend(MSE(ann_all_y, all_data_y))
                    timeElapsedAvgsFunctions.append(ann_end - ann_start)
                    
                    text_file.write("\n\nFunction Class: %s \n" % functionIndex)
                    text_file.write("Function: %s \n" % functions[functionIndex])
                    text_file.write("k = %s, l = %s, m = %s, p = %s, q = %s \n" % (k, l, m, p, q))
                    text_file.write("Activation Function: %s \n" % activationFunction)
                    text_file.write("Minimization Method: %s \n" % minimizationMethod)
                    text_file.write("Number of Neurons: %s \n" % numberNeurons)
                    text_file.write("Mean-Squared-Error: %s \nTime Elapsed: %s \n" % (MSE(ann_all_y, all_data_y), ann_end - ann_start))
                text_file.write("\n\n\n\n\n Average For This NumberNeurons: \n MSE: %s \n Time Elapsed: %s" % (sum(MSEsAvgsNeurons)/len(MSEsAvgsNeurons), sum(timeElapsedAvgsNeurons)/len(timeElapsedAvgsNeurons)))
                MSEsAvgsNeurons = []
                timeElapsedsAvgsNeurons = []
            text_file.write("\n\n\n\n\n Average For This Solver: \n MSE: %s \n Time Elapsed: %s \n\n\n\n\n" % (sum(MSEsAvgsSolvers)/len(MSEsAvgsSolvers), sum(timeElapsedAvgsSolvers)/len(timeElapsedAvgsSolvers)))
            MSEsAvgsSolvers = []
            timeElapsedAvgsSolvers = []
        text_file.write("\n\n\n\n\n Average For This Activation Function: \n MSE: %s \n Time Elapsed: %s \n\n\n\n\n" % (sum(MSEsAvgsAFunctions)/len(MSEsAvgsAFunctions), sum(timeElapsedAvgsAFunctions)/len(timeElapsedAvgsAFunctions)))
        MSEsAvgsAFunctions = []
        timeElapsedAvgsAFunctions = []
    text_file.write("\n\n\n\n\n Average For This Function: \n MSE: %s \n Time Elapsed: %s \n\n\n\n\n" % (sum(MSEsAvgsFunctions)/len(MSEsAvgsFunctions), sum(timeElapsedAvgsFunctions)/len(timeElapsedAvgsFunctions)))
    MSEsAvgsFunctions = []
    timeElapsedAvgsFunctions = []
total_end=time.time()
text_file.write("\n\n\n\n\nTotal Program Runtime: %s seconds" % (total_end - total_start))
