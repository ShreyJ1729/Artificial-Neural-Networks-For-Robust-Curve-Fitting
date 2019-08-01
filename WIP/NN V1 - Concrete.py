from sklearn.neural_network import MLPRegressor
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pandas import ExcelWriter
from pandas import ExcelFile

all_data = pd.read_excel('C:/Users/shrey/OneDrive/Desktop/Concrete_Data.xls', sheet_name='Sheet1')

input1 = np.array(all_data['Cement (component 1)(kg in a m^3 mixture)']).reshape(-1, 1)
input2 = np.array(all_data['Blast Furnace Slag (component 2)(kg in a m^3 mixture)']).reshape(-1, 1)
input3 = np.array(all_data['Fly Ash (component 3)(kg in a m^3 mixture)']).reshape(-1, 1)
input4 = np.array(all_data['Water  (component 4)(kg in a m^3 mixture)']).reshape(-1, 1)
input5 = np.array(all_data['Superplasticizer (component 5)(kg in a m^3 mixture)']).reshape(-1, 1)
input6 = np.array(all_data['Coarse Aggregate  (component 6)(kg in a m^3 mixture)']).reshape(-1, 1)
input7 = np.array(all_data['Fine Aggregate (component 7)(kg in a m^3 mixture)']).reshape(-1, 1)
input8 = np.array(all_data['Age (day)']).reshape(-1, 1)
output = np.array(all_data['Concrete compressive strength(MPa, megapascals) ']).ravel()


nn = MLPRegressor(hidden_layer_sizes=(70, 70, 70), 
                  activation='tanh', solver='adam', max_iter=2000, verbose = True)

nn.fit(np.array([input1, input2, input3, input4, input5, input6, input7, input8]).reshape(1030, -1), output)

print(nn.predict(np.array([540, 0, 0, 162, 2.5, 1040, 676, 28]).reshape(-1, 8)))

input()
