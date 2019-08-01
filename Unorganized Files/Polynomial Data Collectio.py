import time
TOTAL_TIME_START=time.time()
import random
import numpy as np
import matplotlib.pyplot as plt
import math

MSEs=[]
Times=[]

MSEsAvg=[]
TimesAvg=[]

functions = ["((k/l)*(x/l)**(k-1) * np.exp(-(x/l)**(k)))", "1/x", "-(x-p)**2+q*x*np.cos(x)+5", "(np.sin(x))**2+np.sqrt(x)", "k**x" , "k*m*x/l", "m*x*np.sin(3*m*x-2)", "(x**k+1)/(k*x)", "np.sqrt(x**2+m)-np.sqrt(k*2*x)", "-m*x**4+k*x**3"];
training_bounds_all=[[1, 10], [0.1, 10], [0.1, 15], [1.6, 6.3], [0, 5.2]]
    text_file=open("Poly Output REAL.txt", "w")
def percent_accuracy(predicted_y_data, actual_y_data):
    percent_accuracy_all=[]
    for i, j in zip(predicted_y_data, actual_y_data):
        if i<j:
            accuracy=100*abs(j-i)/j
        else:
            accuracy=100*abs(i-j)/i
        while accuracy > 100:
            accuracy -= 100
        percent_accuracy_all.append(abs(accuracy))
    average_accuracy=100-sum(percent_accuracy_all)/len(percent_accuracy_all)
    if average_accuracy < -50:
        average_accuracy = abs(average_accuracy)
    if average_accuracy <0 and average_accuracy > -50:
        average_accuracy += 100
    while average_accuracy > 100:
        average_accuracy -= 100
    return average_accuracy

def MSE_poly(test_data):
    cost=[(i[1]-poly(i[0]))**2 for i in test_data]
    return sum(cost)/(2*len(cost))

for q in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
    for w in range(30):
        for k in [0, 1, 2, 3, 4]:
            text_file.write("Poly %d: \n" % q)
            start_time=time.time()
            def g(x):
                return eval(functions[k])

            def f(x):
                return (g(x) + np.random.normal(g(x), abs(g(x)/10)))/2

            training_bounds = training_bounds_all[k]
            num_training_samples = 15

            training_data = [(i, f(i)) for i in np.linspace(training_bounds[0], training_bounds[1], num_training_samples)]
            training_data_ = training_data
            training_data_x = [i[0] for i in training_data]
            training_data_y = [i[1] for i in training_data]

            x=np.linspace(training_bounds[0], training_bounds[1], 100)
            general_trend_y=[g(i) for i in x]
            actual_y=f(x)
            poly = np.poly1d(np.polyfit(training_data_x, training_data_y, q))
            predicted_y=[poly(i) for i in x]
    
            general_trend_tuples=[(i, g(i)) for i in np.linspace(training_bounds[0], training_bounds[1], num_training_samples)]
            
            end_time=time.time()
            
            text_file.write("NEW FUNCTION: %s \n" % k)
            text_file.write("MSE: %s \nTIME: %s \n" % (MSE_poly(general_trend_tuples), (end_time - start_time)))
            MSEs.append(MSE_poly(general_trend_tuples))
            Times.append(end_time-start_time)
            text_file.write("\n \n")
            print(PercentAccuracies)
            print(sum(MSEs)/len(MSEs))
            print(sum(PercentAccuracies)/len(PercentAccuracies))
        MSEsAvg.append(sum(MSEs)/len(MSEs))
        TimesAvg.append(sum(Times)/len(Times))
text_file.write(" MSE AVG: [")
for i in MSEsAvg:
    text_file.write(" %s ," % i)
text_file.write("] \n \n")

text_file.write(" TIMES AVG: [")
for i in TimesAvg:
    text_file.write(" %s ," % i)
text_file.write("] \n \n")


text_file.write("\n \n \n \n \n Total Program Runtime: %s" % (time.time()-TOTAL_TIME_START))


text_file.close()

