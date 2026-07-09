import numpy as np
from scipy import signal


def butter_lowpass_filter(data, cutoff, fs, order):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    # Get the filter coefficients 
    sos = signal.butter(order, normal_cutoff, btype='low', analog=False, output='sos')
    y = signal.sosfiltfilt(sos, data)
    return y

def butter_bandpass_filter(data, cutoff, fs, order):
    nyq = 0.5 * fs
    normal_cutoff = np.array(cutoff) / nyq
    # Get the filter coefficients 
    sos = signal.butter(order, normal_cutoff, btype='bandpass', analog=False, output='sos')
    y = signal.sosfiltfilt(sos, data)
    return y

    