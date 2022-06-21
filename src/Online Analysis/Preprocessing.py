from scipy.signal import butter, lfilter

from ConfigUnicorn import *

import pywt


def remove_motion_artifacts(signalnp, wavelet='sym5', level=6, ):
    def iqr(sig):
        std = np.std(sig)
        return std

    for i in range(8):
        sig = signalnp[:, i]

        coeff = pywt.wavedec(sig, wavelet, level=level)
        # print(10*iqr(sig))
        thresh = 7 * iqr(sig)
        fto = 4
        coeff[:fto] = (pywt.threshold(c, thresh, mode='less', substitute=0) for c in coeff[:fto])
        coeff[:fto] = (pywt.threshold(c, -thresh, mode='greater', substitute=0) for c in coeff[:fto])
        clean_sig = pywt.waverec(coeff, wavelet, mode='smooth')

        # print(clean_sig)
        signalnp[:, i] = clean_sig[:len(sig)]



def notch_Filter(sample):
    from scipy import signal
    b_notch, a_notch = signal.iirnotch(NOTCH_FREQ, QUALITY_FACTOR, SAMPLING_FREQUENCY)
    outputSignal = signal.filtfilt(b_notch, a_notch, sample)
    b_notch, a_notch = signal.iirnotch(6, QUALITY_FACTOR, SAMPLING_FREQUENCY)
    outputSignal = signal.filtfilt(b_notch, a_notch, outputSignal)
    return outputSignal


def butter_bandpass(lowcut, highcut, fs):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(N=4, Wn=[low, high], btype='band')
    return b, a

def band_Pass_filter(sample):
    b, a = butter_bandpass(1, 10, 250)
    y = lfilter(b, a, sample)
    return y


def preprocess_data(raw):
    for i in range(NUMBER_OF_CHANNELS):
        notch_Filter(np.copy(raw[:, i]))
        band_Pass_filter(np.copy(raw[:, i]))

    avarageSignal = np.mean(raw, axis=1)
    for i in range(NUMBER_OF_CHANNELS):
        raw[:, i] -= avarageSignal

    for i in range(NUMBER_OF_CHANNELS):
        mean = raw[:, i].mean()
        raw[:, i] = (raw[:, i] - mean)
    remove_motion_artifacts(raw)
