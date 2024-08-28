import h5py
from os.path import dirname, basename, splitext, join
from os import listdir
import pandas as pd
import numpy as np
import seaborn
import matplotlib.pyplot as plt
import scipy
from scipy.signal import butter, filtfilt
from sklearn import preprocessing
from functions import *

#Path is datafile, BEHAVIOUR_PATH is excel notes, SEC_HZ is acquisition frequency (40 frames per second)
PATH = r''
SEC_HZ = 40
BASE_LENGTH = 120
REC_LENGTH = 4 * 60
FRAME_LENGTH = REC_LENGTH * SEC_HZ
SMALL_SIZE = 12
MEDIUM_SIZE = 20
BIGGER_SIZE = 36
T = REC_LENGTH         # Sample Period
fs = SEC_HZ       # sample rate, Hz
cutoff = 3      # desired cutoff frequency of the filter, Hz ,      slightly higher than actual 1.2 Hz
nyq = 0.5 * fs  # Nyquist Frequency
order = 2       # sin wave can be approx represented as quadratic
n = int(T * fs)  # total number of samples


def get_photometry_data(p):
    AIn = 'AIn-1'
    with h5py.File(p, 'r') as h:
        list_keys = [key for key in h['Traces'].keys()]
        if 'Console' in list_keys:
            is_3_in_list = [1 for key in list_keys if '3' in key]
            if len(is_3_in_list) > 0:
                AIn = 'AIn-2'
            alog1 = np.array(h['Traces']['Console'][f'{AIn} - Dem (AOut-1)'][f'{AIn} - Dem (AOut-1)'])
            alog2 = np.array(h['Traces']['Console'][f'{AIn} - Dem (AOut-2)'][f'{AIn} - Dem (AOut-2)'])
            analog1 = [item for item in alog1 if not(math.isnan(item)) == True]
            analog2 = [item for item in alog2 if not (math.isnan(item)) == True]

        else:
            is_3_in_list = [1 for key in list_keys if '3' in key]
            if len(is_3_in_list) > 0:
                AIn = 'AIn-2'
            alog1 = np.array(h['Traces'][f'{AIn} - Dem (AOut-1)'][f'{AIn} - Dem (AOut-1)'])
            alog2 = np.array(h['Traces'][f'{AIn} - Dem (AOut-2)'][f'{AIn} - Dem (AOut-2)'])
            analog1 = [item for item in alog1 if not(math.isnan(item)) == True]
            analog2 = [item for item in alog2 if not (math.isnan(item)) == True]
    return analog1, analog2

def scaling_factor(analog1, analog2):
    baseline_array1 = analog1[0: BASE_LENGTH * SEC_HZ]
    mean_baseline_array1 = sum(baseline_array1) / (BASE_LENGTH*SEC_HZ)
    baseline_array2 = analog2[0: BASE_LENGTH * SEC_HZ]
    mean_baseline_array2 = sum(baseline_array2) / (BASE_LENGTH*SEC_HZ)
    correction_factor = mean_baseline_array1/mean_baseline_array2
    analog1 = np.array(analog1)
    analog2 = np.array(analog2)
    array = analog1 - (analog2 * correction_factor) + np.nanmedian(analog2 * correction_factor)
    return array


def df(array):
    array = np.asarray(array)
    baseline_array = array[0: BASE_LENGTH * SEC_HZ]
    mean_baseline_array = np.nansum(baseline_array) / (BASE_LENGTH * SEC_HZ)
    array_df = (array - mean_baseline_array) / mean_baseline_array
    array_df_nonan = array_df[~np.isnan(array_df)]
    array_rec = array_df_nonan[0: REC_LENGTH*SEC_HZ]
    return array_rec



def butter_lowpass_filter(data, cutoff, fs, order):
    normal_cutoff = cutoff / nyq
    # Get the filter coefficients
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = filtfilt(b, a, data)
    return y


def linegraph(array, title):
    plt.rc('font', size=MEDIUM_SIZE)  # controls default text sizes
    plt.rc('axes', titlesize=MEDIUM_SIZE)  # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
    plt.rc('xtick', labelsize=MEDIUM_SIZE)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=MEDIUM_SIZE)  # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)  # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
    plt.plot(array, color='black',  label=title)
    plt.axvline((BASE_LENGTH*SEC_HZ), color='black', linestyle='--')
    plt.title(str(title))
    plt.xlabel('Time (min)')
    plt.ylabel('dF/F')
    plt.yticks(np.arange(-0.1, 0.1001, step=0.05), rotation='horizontal')
    plt.xticks(np.arange(0, FRAME_LENGTH + 1, step=FRAME_LENGTH / 2), ['-2', '0', '2'], rotation='horizontal')
    plt.savefig(join(PATH, title + '.png'))
    # plt.show()
    plt.close()


if __name__ == "__main__":
    path_list = [join(PATH, file) for file in listdir(PATH)]
    for p in path_list:
        title = str(splitext(basename(p))[0])
        analog1, analog2 = get_photometry_data(p)
        array = scaling_factor(analog1, analog2)
        array_df = df(array)[0:REC_LENGTH*SEC_HZ]
        array_df_filter = butter_lowpass_filter(array_df, 0.3, SEC_HZ, 2)
        linegraph(array_df_filter, title)
        np.savetxt(f'{PATH+title}.csv', array_df_filter, fmt='%.5f', delimiter=",")
