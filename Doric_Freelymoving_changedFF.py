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

#Path is datafile, BEHAVIOUR_PATH is excel notes, SEC_HZ is acquisition frequency
PATH1 = r'C:\Users\mminere\Desktop\Fiber Photometry\Data\POMC neurons\Fast_Chow\Doric Files'
PATH2 = r'C:\Users\mminere\Desktop\Fiber Photometry\Data\POMC-PVT\Data Recs\Refed\Doric Files\Fast_Chow'
BEHAVIOR_PATH = r'C:\Users\mminere\Desktop\Fiber Photometry\Annotations\PVTµOPR\OPRM1 recordings annotated.xlsx'
PLOT_TITLE = 'POMC neuron vs POMC-PVT axons'
SEC_HZ = 40
BASE_LENGTH = 300
START_REMOVE = 0
MANIPULATION = BASE_LENGTH - START_REMOVE
REC_LENGTH = 600
SMALL_SIZE = 12
MEDIUM_SIZE = 16
BIGGER_SIZE = 20
T = REC_LENGTH  # Sample Period
fs = SEC_HZ     # sample rate, Hz
cutoff = 3      # desired cutoff frequency of the filter, Hz ,      slightly higher than actual 1.2 Hz
nyq = 0.5 * fs  # Nyquist Frequency
order = 2       # sin wave can be approx represented as quadratic
n = int(T * fs) # total number of samples

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
            analog1 = [item for item in alog1 if not (math.isnan(item)) == True]
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
    baseline_array1 = analog1[START_REMOVE * SEC_HZ: BASE_LENGTH * SEC_HZ]
    mean_baseline_array1 = sum(baseline_array1) / ((BASE_LENGTH-START_REMOVE)*SEC_HZ)
    baseline_array2 = analog2[(START_REMOVE) * SEC_HZ: BASE_LENGTH * SEC_HZ]
    mean_baseline_array2 = sum(baseline_array2) / ((BASE_LENGTH-START_REMOVE)*SEC_HZ)
    correction_factor = mean_baseline_array1/mean_baseline_array2
    analog1 = np.array(analog1)
    analog2 = np.array(analog2)
    array = analog1 - (analog2 * correction_factor) + np.nanmedian(analog2 * correction_factor)
    return array


def butter_lowpass_filter(data, cutoff, fs, order):
    normal_cutoff = cutoff / nyq
    # Get the filter coefficients
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = filtfilt(b, a, data)
    return y


def df(array):
    array = np.asarray(array)
    baseline_array = array[0: BASE_LENGTH * SEC_HZ]
    mean_baseline_array = np.nansum(baseline_array) / (BASE_LENGTH * SEC_HZ)
    array_df = (array - mean_baseline_array) / mean_baseline_array
    array_df_nonan = array_df[~np.isnan(array_df)]
    return array_df_nonan


def z_score(array):
    baseline_array = array[0:MANIPULATION*SEC_HZ]
    mean = np.nanmean(baseline_array)
    sd = np.nanstd(baseline_array)
    array_z = (array - mean) / sd
    return array_z


def event_array_path1(path):
    path_list = [join(path, file) for file in listdir(path)]
    id_axis = []
    base_amp_quant = []
    final_amp_quant = []
    data_arrays_df = np.array([])
    for p in path_list:
        id = str(splitext(basename(p))[0])
        id_axis.append(id)
        analog1, analog2 = get_photometry_data(p)
        array = scaling_factor(analog1, analog2)
        array_df = df(array)
        array_df_filtered = butter_lowpass_filter(array_df, 0.5, SEC_HZ, 2)[0:REC_LENGTH*SEC_HZ]
        base_amp = array_df_filtered[(BASE_LENGTH-30) * SEC_HZ :BASE_LENGTH * SEC_HZ]
        final_amp = array_df_filtered[(REC_LENGTH-30) * SEC_HZ: REC_LENGTH * SEC_HZ]
        base_amp_avg = base_amp.mean()
        final_amp_avg = final_amp.mean()
        base_amp_quant.append(base_amp_avg)
        final_amp_quant.append(final_amp_avg)
        if len(data_arrays_df) == 0:
            data_arrays_df = array_df_filtered
        else:
            data_arrays_df = np.vstack((data_arrays_df, array_df_filtered))
    data_arrays_matrix_df = pd.DataFrame(data_arrays_df)
    return data_arrays_matrix_df, base_amp_quant, final_amp_quant


def event_array_path2(path):
    path_list = [join(path, file) for file in listdir(path)]
    id_axis = []
    base_amp_quant = []
    final_amp_quant = []
    data_arrays_df = np.array([])
    for p in path_list:
        id = str(splitext(basename(p))[0])
        id_axis.append(id)
        analog1, analog2 = get_photometry_data(p)
        array = scaling_factor(analog1, analog2)[120*SEC_HZ:660*SEC_HZ]
        array_df = df(array)
        array_df_filtered = butter_lowpass_filter(array_df, 0.5, SEC_HZ, 2)[0:REC_LENGTH*SEC_HZ]
        base_amp = array_df_filtered[(BASE_LENGTH-30) * SEC_HZ :BASE_LENGTH * SEC_HZ]
        final_amp = array_df_filtered[(REC_LENGTH-30) * SEC_HZ: REC_LENGTH * SEC_HZ]
        base_amp_avg = base_amp.mean()
        final_amp_avg = final_amp.mean()
        base_amp_quant.append(base_amp_avg)
        final_amp_quant.append(final_amp_avg)
        if len(data_arrays_df) == 0:
            data_arrays_df = array_df
        else:
            data_arrays_df = np.vstack((data_arrays_df, array_df))
    data_arrays_matrix_df = pd.DataFrame(data_arrays_df)
    return data_arrays_matrix_df, base_amp_quant, final_amp_quant


def linegraph(data1, ylabel, title):
    data1_mean = data1.mean()
    # data1_min = round(data1_mean.min(), 2) - 0.1
    # data1_max = round(data1_mean.max(), 2) + 0.1
    sem1 = scipy.stats.sem(data1)
    plt.rc('font', size=MEDIUM_SIZE)  # controls default text sizes
    plt.rc('axes', titlesize=MEDIUM_SIZE)  # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
    plt.rc('xtick', labelsize=MEDIUM_SIZE)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=MEDIUM_SIZE)  # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)  # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
    plt.plot(data1_mean, color='black',  label='')
    plt.fill_between(range(len(data1_mean)), data1_mean - sem1, data1_mean + sem1, alpha=0.25)
    # plt.legend(loc="upper right")
    plt.axvline(x=MANIPULATION*SEC_HZ, color='black', linestyle='--')
    plt.title(title)
    plt.xlabel('Time (min)')
    plt.ylabel(ylabel)
    plt.yticks(np.arange(-0.2, 0.1+0.01, step=0.1), rotation='horizontal')
    plt.xticks(np.arange(0, REC_LENGTH + 1, step=REC_LENGTH / 3), ['0', '5', '10', '15'], rotation='horizontal')
    plt.show()
    plt.close()
    return data1_mean, sem1


def linegraph_2data(data1, data2, ylabel):
    data1_mean = data1.mean()
    data2_mean = data2.mean()
    data1_min = round(data1_mean.min(), 2) -0.1
    data1_max = round(data1_mean.max(), 2) +0.1
    sem1 = scipy.stats.sem(data1)
    sem2 = scipy.stats.sem(data2)
    plt.rc('font', size=MEDIUM_SIZE)  # controls default text sizes
    plt.rc('axes', titlesize=MEDIUM_SIZE)  # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
    plt.rc('xtick', labelsize=MEDIUM_SIZE)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=MEDIUM_SIZE)  # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)  # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
    plt.plot(data1_mean, color='black',  label='Saline')
    plt.fill_between(range(len(data1_mean)), data1_mean - sem1, data1_mean + sem1, alpha=0.25)
    plt.plot(data2_mean, color='blue',  label='CNO')
    plt.fill_between(range(len(data2_mean)), data2_mean - sem2, data2_mean + sem2, alpha=0.25)
    plt.legend(loc="upper right")
    plt.axvline(x=MANIPULATION*SEC_HZ, color='black', linestyle='--')
    plt.title(PLOT_TITLE)
    plt.xlabel('Time (min)')
    plt.ylabel(ylabel)
    plt.yticks(np.arange(-0.2, 0.1+0.01, step=0.1), rotation='horizontal')
    plt.xticks(np.arange(0, REC_LENGTH + 1, step=REC_LENGTH / 3), ['0', '5', '10', '15'], rotation='horizontal')
    plt.show()
    plt.savefig(PATH1)
    plt.close()


if __name__ == "__main__":
    df_matrix_sal, base_sal, final_sal = event_array_path1(PATH1)
    df_matrix_cno, base_cno, final_cno = event_array_path2(PATH2)
    sal_mean, sal_sem = linegraph(df_matrix_sal, 'dF/F', 'Saline')
    cno_mean, cno_sem = linegraph(df_matrix_cno, 'dF/F', 'CNO')
    linegraph_2data(df_matrix_sal, df_matrix_cno, 'dF/F')
    data_sal = np.column_stack([sal_mean, sal_sem])
    data_cno = np.column_stack([cno_mean, cno_sem])
    np.savetxt(f'{PATH1}.csv', data_sal, fmt=['%.5f', '%.5f'], delimiter=";")
    np.savetxt(f'{PATH2}.csv', data_cno, fmt=['%.5f', '%.5f'], delimiter=";")
    print(base_sal)             #quantification of baseline signal prior to intervention
    print(final_sal)            #quantification of signal end of recording
    print(base_cno)             #quantification of baseline signal prior to intervention
    print(final_cno)            #quantification of signal end of recording