import matplotlib.pyplot as plt

from config import *

import numpy as np
from scipy.signal import butter, lfilter


def butter_bandpass(lowcut, highcut, fs):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(N=4, Wn=[low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs):
    b, a = butter_bandpass(lowcut, highcut, fs)
    y = lfilter(b, a, data)
    # f_signal = rfft(data)

    # W = rfftfreq(data.size, d=1 / freq)
    # cut_f_signal = f_signal.copy()
    # cut_f_signal[(W > highcut_freq)] = 0
    # cut_f_signal[(W < lowcut_freq)] = 0
    #
    # cut_signal = irfft(cut_f_signal)
    # y = cut_signal
    return y

def draw(arr,m):
    x_axis = np.array(range(arr.shape[0]))+4000
    plt.plot(x_axis, arr, color='#188038')
    plt.title(m)
    plt.xlabel('Sample')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.tight_layout()
    plt.show()

def preprocessing_signal_train_test(train, test):
    plt.figure(figsize=(12, 6), dpi=200)


    draw(train[:,:,10].ravel()[4000:6000],"Raw")
    train_res = train
    test_res = test

    train_avarageSignal = np.zeros(train.shape[0] * train.shape[1])

    for channel in range(NUMBER_OF_CHANNELS):
        train_avarageSignal += train[:, :, channel].ravel()
    train_avarageSignal = train_avarageSignal / NUMBER_OF_CHANNELS

    test_avarageSignal = np.zeros(test.shape[0] * test.shape[1])
    for channel in range(NUMBER_OF_CHANNELS):
        test_avarageSignal += test[:, :, channel].ravel()
    test_avarageSignal = test_avarageSignal / NUMBER_OF_CHANNELS

    for channel in CHANNELS:
        # ButterWorth Filter
        train_signal = train[:, :, channel].ravel() - train_avarageSignal

        if(channel == 10):
            plt.figure(figsize=(12, 6), dpi=200)

            draw(train_signal[4000:6000], "CAR")
        filtered_train_signal = butter_bandpass_filter(train_signal, LOWCUT_FREQ, HIGHCUT_FREQ, FREQUENCY)
        test_signal = test[:, :, channel].ravel() - test_avarageSignal
        filtered_test_signal = butter_bandpass_filter(test_signal, LOWCUT_FREQ, HIGHCUT_FREQ, FREQUENCY)
        if(channel == 10):
            plt.figure(figsize=(12, 6), dpi=200)

            draw(filtered_train_signal[4000:6000], "Band-Pass")

        # Normalizing
        mean = filtered_train_signal.mean()
        std = filtered_train_signal.std()
        normalized_train_signal = (filtered_train_signal - mean) / std
        normalized_test_signal = (filtered_test_signal - mean) / std

        # scaler.fit(filtered_train_signal)
        #
        # filtered_train_signal = scaler.transform(filtered_train_signal)
        # filtered_test_signal = scaler.transform(filtered_test_signal)
        if(channel == 10):
            plt.figure(figsize=(12, 6), dpi=200)

            draw(normalized_train_signal[4000:6000], "Normalized")
        train_res[:, :, channel] = normalized_train_signal.reshape(train.shape[0], train.shape[1])
        test_res[:, :, channel] = normalized_test_signal.reshape(test.shape[0], test.shape[1])
    return train_res, test_res
