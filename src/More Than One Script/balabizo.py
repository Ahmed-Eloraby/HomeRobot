# import numpy as np
# import matplotlib.pyplot as plt
#
# my_data = np.genfromtxt('data.csv', delimiter=',')
# plt.figure(figsize=(15, 13), dpi=80)
#
# x = range(my_data.shape[0])
# f, axis = plt.subplots(8)
# f.set_figheight(50)
# f.set_figwidth(15)
# print(axis.shape)
# for i in range(8):
#     axis[i].plot(x,my_data[:,i])
#     axis[i].set_title(f"Channel {i+1}")
# plt.subplots_adjust(wspace=0.4,
#                     hspace=0.9)
# plt.show()
#
# from ConfigUnicorn import *
# from scipy.signal import butter, lfilter
#
#
# def notch_Filter(sample):
#     from scipy import signal
#     b_notch, a_notch = signal.iirnotch(NOTCH_FREQ, QUALITY_FACTOR, SAMPLING_FREQUENCY)
#     outputSignal = signal.filtfilt(b_notch, a_notch, sample)
#     return outputSignal
#
#
# def butter_bandpass(lowcut, highcut, fs):
#     nyq = 0.5 * fs
#     low = lowcut / nyq
#     high = highcut / nyq
#     b, a = butter(N=4, Wn=[low, high], btype='band')
#     return b, a
#
#
# def band_Pass_filter_raw(sample):
#     b, a = butter_bandpass(0.1, 60, 250)
#     y = lfilter(b, a, sample)
#     return y
#
#
#
# raw_data = np.copy(my_data[15000:,:8])
# filtered_data = np.copy(raw_data)
# for i in range(NUMBER_OF_CHANNELS):
#     y = band_Pass_filter_raw(np.copy(raw_data[:,i]))
#     filtered_data[:,i] = y
#
# x = range(raw_data.shape[0])
# f, axis = plt.subplots(8)
# f.set_figheight(50)
# f.set_figwidth(15)
# print(axis.shape)
# for i in range(8):
#     axis[i].plot(x,raw_data[:,i])
#     axis[i].set_title(f"Channel {i+1}")
# plt.subplots_adjust(wspace=0.4,
#                     hspace=0.9)
# plt.show()
#
#
# x= range(filtered_data.shape[0])
# f, axis = plt.subplots(8)
# f.set_figheight(200)
# f.set_figwidth(15)
# print(axis.shape)
# for i in range(8):
#     axis[i].plot(x,filtered_data[:,i])
#     axis[i].set_title(f"Channel {i+1}")
# plt.subplots_adjust(wspace=0.4,
#                     hspace=0.9)
# plt.show()

from os import mkdir, path
from ConfigUnicorn import *

from scipy.io import savemat

name =NAME
trainortest = TRAIN_OR_TEST

if not path.isdir(f"{DATA_PATH}/{name}"):
    mkdir(f"{DATA_PATH}/{name}")
if not path.isdir(f"{DATA_PATH}/{name}/{trainortest}"):
    mkdir(f"{DATA_PATH}/{name}/{trainortest}")
import os
count = 0

# Iterate directory
for path in os.listdir(f"{DATA_PATH}/{name}/{trainortest}"):
    # check if current path is a file
    count += 1
z =np.array([1,2,3])
savemat(f"{DATA_PATH}/{name}/{trainortest}/bci{('{:02d}'.format(count+1))}.mat", {'Samples': z})
