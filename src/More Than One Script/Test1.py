import h5py
import numpy as np

hf = h5py.File('data.h5', 'w')
d2 = np.array([["balabizo"],["s"]])
hf.create_dataset('dataset_2', data=d2)

hf.close()
