from scipy.signal import butter, lfilter

from ConfigUnicorn import *


def notch_Filter(sample):
    from scipy import signal
    b_notch, a_notch = signal.iirnotch(NOTCH_FREQ, QUALITY_FACTOR, SAMPLING_FREQUENCY)
    outputSignal = signal.filtfilt(b_notch, a_notch, sample)
    return outputSignal


def butter_bandpass(lowcut, highcut, fs):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(N=4, Wn=[low, high], btype='band')
    return b, a


def band_Pass_filter_raw(sample):
    b, a = butter_bandpass(0.1, 60, SAMPLING_FREQUENCY)
    y = lfilter(b, a, sample)
    return y


def band_Pass_filter(sample):
    b, a = butter_bandpass(LOWCUT_FREQ, HIGHCUT_FREQ, SAMPLING_FREQUENCY)
    y = lfilter(b, a, sample)
    return y


def preprocessing_Raw(data):
    for i in range(NUMBER_OF_CHANNELS):
        y = band_Pass_filter_raw(data[i])
        z = notch_Filter(y)
        data[i] = z


def preprocessing(data, is_train):
    # Common Average Reference
    avarageSignal = np.mean(data, axis=1)
    for i in range(NUMBER_OF_CHANNELS):
        data[i] -= avarageSignal

    # Band_Pass_Filter
    for i in range(NUMBER_OF_CHANNELS):
        filtered_signal = band_Pass_filter(data[i])
        data[i] = filtered_signal

    # Standardization
    if is_train:
        means = []
        stds = []
        for i in range(NUMBER_OF_CHANNELS):
            mean = data[i].mean()
            std = data[i].std()
            means.append(mean)
            stds.append(std)
            data[i] = (data[i] - mean) / std
            with open(f'./experiment/{NAME}/StandardizationValues.npy', 'wb') as f:
                np.save(f, np.array(means))
                np.save(f, np.array(stds))

    else:
        with open('./experiment/{NAME}/StandardizationValues.npy', 'rb') as f:
            means = np.load(f)
            stds = np.load(f)
        for i in range(NUMBER_OF_CHANNELS):
            data[i] = (data[i] - means[i]) / stds[i]
