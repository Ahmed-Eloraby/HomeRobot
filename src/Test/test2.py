import numpy as np
import pandas as pd
data = pd.read_csv("data.csv",header = None).to_numpy()
print(data)
# my_data = np.genfromtxt('data.csv',delimiter=',')
# print(my_data)
# initial_time = my_data[0,0]