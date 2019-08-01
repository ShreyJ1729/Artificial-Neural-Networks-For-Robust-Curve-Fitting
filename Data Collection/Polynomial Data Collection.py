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

functions = ["((k/l)*(x/l)**(k-1) * np.exp(-(x/l)**(k)))", "1/x", "-(x-p)**2+q*x*np.cos(x)+5", "(np.sin(x))**2+np.sqrt(x)", "k**x" , "k*m*x/l", "m*x*np.sin(3*m*x-2)", "(x**k+1)/(k*x)", "np.sqrt(x**2+m)-np.sqrt(k*2*x)", "-m*x**4+k*x**3"]
text_file=open("Output_poly.txt", "w")

def MSE_poly(test_data):
    cost=[(i[1]-poly(i[0]))**2 for i in test_data]
    return sum(cost)/(len(cost))

for a in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
    for w in range(50):
        for y in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]:
            k=random.randint(1, 5);
            l=random.randint(1, 5);
            m = (random.randint(1, 6))/3;
            p = (random.randint(300, 400)/100);
            q = (random.randint(100, 200)/100);
            text_file.write("Poly %d: \n" % a)
            start_time=time.time()
            def g(x):
                return eval(functions[y])

            def f(x):
                return (g(x) + np.random.normal(g(x), abs(g(x)/10)))/2

            training_bounds = [0, 6]
            num_training_samples = 20

            training_data = [(i, f(i)) for i in np.linspace(training_bounds[0], training_bounds[1], num_training_samples)]
            training_data_ = training_data
            training_data_x = [i[0] for i in training_data]
            training_data_y = [i[1] for i in training_data]

            x=np.linspace(training_bounds[0], training_bounds[1], 100)
            general_trend_y=[g(i) for i in x]
            actual_y=f(x)
            poly = np.poly1d(np.polyfit(training_data_x, training_data_y, a))
            predicted_y=[poly(i) for i in x]
    
            general_trend_tuples=[(i, g(i)) for i in np.linspace(training_bounds[0], training_bounds[1], num_training_samples)]
            
            end_time=time.time()
            
            text_file.write("Function Class Chosen: %s \n" % y)
            text_file.write("MSE: %s \nTIME: %s \n" % (MSE_poly(general_trend_tuples), (end_time - start_time)))
            text_file.write("k = %s, l = %s, m = %s, p = %s, q = %s \n" % (k, l, m, p, q))
            MSEs.append(MSE_poly(general_trend_tuples))
            Times.append(end_time-start_time)
            text_file.write("\n \n")
        MSEsAvg.append(sum(MSEs)/len(MSEs))
        TimesAvg.append(sum(Times)/len(Times))
text_file.write(" MSE AVG: [")
for i in MSEsAvg:
    text_file.write(" %s ," % i)
text_file.write("] \n \n")

text_file.write(" Time Elapsed AVG: [")
for i in TimesAvg:
    text_file.write(" %s ," % i)
text_file.write("] \n \n")


text_file.write("\n \n \n \n \n Total Program Runtime: %s" % (time.time()-TOTAL_TIME_START))


text_file.close()


