import numpy as np
from scipy import signal
x= np.array([1,2,3,4,5,6,7,8,1,2,3,4,5,6,7,8,1,2,3,4,5,6,7,8,1,2,3,4,5,6,7,8,0])
print(len(x))
print(len(signal.decimate(x,4)))
print(x[1::3])