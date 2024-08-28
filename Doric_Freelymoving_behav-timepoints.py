import h5py
from os.path import dirname, basename, splitext, join
from os import listdir
import pandas as pd
import numpy as np
import seaborn
import matplotlib.pyplot as plt
import scipy
from sklearn import preprocessing
from functions import *

#Change path to datafile, behaviour path to excel notes, array timing with time period prior and post baseline, event line to frame of intervention
#Determine x-axis tick length and update ticks, choose time in (s) or (min)
#Change plot title and ID split in behaviour timepoints function to analysis group name

#Path is datafile, BEHAVIOUR_PATH is excel notes, SEC_HZ is acquisition frequency (40 frames per second)
PATH = r'C:\Users\mminere\Desktop\Fiber Photometry\Final Dataset POMC-PVT\Data Recs\Refed\Fast-refeed-HSD_HSD'
BEHAVIOR_PATH = r'C:\Users\mminere\Desktop\Fiber Photometry\Annotations\POMC-PVT\POMC-PVT all rec notes.xlsx'
SEC_HZ = 40
SMALL_SIZE = 12
MEDIUM_SIZE = 21
BIGGER_SIZE = 32


def line_graph(array_df, event):
    df = pd.DataFrame(array_df)
    array_df_mean = df.mean()
    sem = scipy.stats.sem(array_df)
    plt.rc('font', size=MEDIUM_SIZE)  # controls default text sizes
    plt.rc('axes', titlesize=MEDIUM_SIZE)  # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
    plt.rc('xtick', labelsize=MEDIUM_SIZE)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=MEDIUM_SIZE)  # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)  # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
    plt.plot(array_df_mean, color='k')
    plt.fill_between(range(len(array_df_mean)), array_df_mean - sem, array_df_mean + sem, alpha=0.25)
    plt.axvline(1200, color='black', linestyle='--')
    plt.title(event)
    plt.xlabel('Time (s)')
    plt.ylabel('dF/F')
    plt.ylim(-(np.max(sem) + np.max(array_df_mean) + 0.1), np.max(sem) + np.max(array_df_mean) + 0.1)
    plt.xticks(np.arange(0, len(array_df[0]) + 1, step=len(array_df[0]) * 0.25), ['0', '15', '30', '45', '60'])
    plt.show()
    return array_df_mean, sem
    # plt.close()


def heatmap_plot(matrix, ID_axis, experiment, event):
    # plt.rc('font', size=MEDIUM_SIZE)  # controls default text sizes
    plt.rc('axes', titlesize=MEDIUM_SIZE)  # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
    plt.rc('xtick', labelsize=MEDIUM_SIZE)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=MEDIUM_SIZE)  # fontsize of the tick labels
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
    sn = seaborn.heatmap(matrix, vmin=-0.05, vmax=0.075)
    caxis = sn.figure.axes[-1]
    caxis.tick_params(labelsize=18)
    plt.rc('legend', fontsize=1)  # legend fontsize
    plt.axvline(x=1200, color='k', linestyle='--')
    plt.title(event)
    # plt.title(experiment+event)
    plt.xlabel('Time (s)')
    plt.ylabel('Mouse ID')
    ytick = list(range(len(matrix)))
    yticks =[x+1 for x in ytick]
    plt.yticks(np.arange(len(matrix)), yticks, rotation='horizontal')
    plt.xticks(np.arange(0,len(matrix.columns)+1, step=len(matrix.columns)*0.25), ['0', '15', '30', '45', '60'], rotation='horizontal')
    plt.show()
    #plt.savefig(join(folder2, 'all_trials.png'))

def behaviour_timepoints():
    ID_timepoints = {}
    ID_sampling = {}
    sheet_name = basename(dirname(PATH))
    df = pd.read_excel(BEHAVIOR_PATH, sheet_name=sheet_name)
    ID_list = df.loc[:, 'ID']
    group_name = PATH.split('\\')[-1]
    for item in ID_list:
        tmp = list(df.loc[df['Experiment'] == group_name,'ID'])
        sub_df = df.loc[df['Experiment'] == group_name,:]
        if item in tmp:
            item_str = str(int(item))
            ID_timepoints[item_str] = {}
            ID_timepoints[item_str]['SEC_HZ'] = int(sub_df.loc[sub_df['ID'] == item, 'SEC_HZ'])
            sec_hz = ID_timepoints[item_str]['SEC_HZ']
            ID_sampling[item_str] = sec_hz
            ID_timepoints[item_str]['pellet in'] = int(sub_df.loc[sub_df['ID'] == item, 'pellet in'] * SEC_HZ)
            ID_timepoints[item_str]['head orient'] = int(sub_df.loc[sub_df['ID'] == item, 'head orient'] * SEC_HZ)
            ID_timepoints[item_str]['start 1'] = int(sub_df.loc[sub_df['ID'] == item, 'Start 1'] * SEC_HZ)
            ID_timepoints[item_str]['end 1'] = int(sub_df.loc[sub_df['ID'] == item, 'End 1'] * SEC_HZ)
    return ID_timepoints, ID_sampling


def main():
    dictionary, sec_hz_dict = behaviour_timepoints()
    path_list = [join(PATH, file) for file in listdir(PATH)]
    # experiment = PATH.split('Refed\\')[1].split('.')[0]
    heatmap_start = np.array([])
    heatmap_end = np.array([])
    heatmap_pellet = np.array([])
    heatmap_head = np.array([])
    ID_axis = []
    full_array = []
    pellet = []
    head = []
    start = []
    end = []

    for p in path_list:
        ID = p.split('Fast-refeed-HSD_HSD\\')[1].split('.')[0]
        ID_axis.append(ID)
        timing_pellet = dictionary[ID]['pellet in']
        timing_head = dictionary[ID]['head orient']
        timing_start = dictionary[ID]['start 1']
        timing_end = dictionary[ID]['end 1']
        analog1, analog2 = get_photometry_data(p, sec_hz_dict[ID])
        array = scaling_factor(analog1, analog2)
        array_df = df_calc(array, timing_pellet)[0:(600*SEC_HZ)]
        array_df_nonan = array_df[~np.isnan(array_df)]

        # array_df_pellet = df_calc(array, timing_pellet)
        # array_df_head = df_calc(array, timing_head)
        # array_df_start = df_calc(array, timing_start)
        # array_df_end = df_calc(array, timing_end)

        pellet_array = array_df[timing_pellet - (30 * SEC_HZ): timing_pellet + (60 * SEC_HZ)]
        head_array = array_df[timing_head - (30 * SEC_HZ): timing_head + (60 * SEC_HZ)]
        start_array = array_df[timing_start - (30 * SEC_HZ): timing_start + (60 * SEC_HZ)]
        end_array = array_df[timing_end - (30 * SEC_HZ): timing_end + (60 * SEC_HZ)]

        array_df_scaling = preprocessing.normalize([array_df_nonan])
        pellet_array_scaling = preprocessing.normalize([pellet_array])
        head_array_scaling = preprocessing.normalize([head_array])
        start_array_scaling = preprocessing.normalize([start_array])
        #end_array_scaling = preprocessing.normalize([end_array])
        array_df_norm = array_df_scaling.squeeze()
        pellet_array_norm = pellet_array_scaling.squeeze()
        head_array_norm = head_array_scaling.squeeze()
        start_array_norm = start_array_scaling.squeeze()
        #end_array_norm = end_array_scaling.squeeze()
        full_array.append(array_df_norm)
        pellet.append(pellet_array_norm)
        head.append(head_array_norm)
        start.append(start_array_norm)
        #end.append(end_array_norm)

        # if len(heatmap_pellet) == 0:
        #     heatmap_pellet = pellet_array_norm
        #     heatmap_head = head_array_norm
        #     heatmap_start = start_array_norm
        #     heatmap_end = end_array_norm
        # else:
        #     heatmap_pellet = np.vstack((heatmap_pellet, pellet_array_norm))
        #     heatmap_head = np.vstack((heatmap_head, head_array_norm))
        #     heatmap_start = np.vstack((heatmap_start, start_array_norm))
        #     heatmap_end = np.vstack((heatmap_end, end_array_norm))


    #Create Heatmap for all recordings in PATH
    # pellet_matrix = pd.DataFrame(heatmap_pellet)
    # head_matrix = pd.DataFrame(heatmap_head)
    # start_matrix = pd.DataFrame(heatmap_start)
    # end_matrix = pd.DataFrame(heatmap_end)
    #
    # heatmap_plot(pellet_matrix, ID_axis, experiment, ' Pellet')
    # heatmap_plot(head_matrix, ID_axis, experiment, ' Head orientation')
    # heatmap_plot(start_matrix, ID_axis, experiment, ' start')
    # heatmap_plot(end_matrix, ID_axis, experiment, ' end')

    #create line plot for all recordings in PATH
    full_array_mean, full_array_sem = line_graph(full_array, pellet)
    pellet_mean, pellet_sem = line_graph(pellet, ' pellet')
    head_mean, head_sem = line_graph(head, ' head orientation')
    start_mean, start_sem = line_graph(start, ' start')
    #end_mean, end_sem = line_graph(end, ' end')

    # save data to csv files
    data_pellet = np.column_stack([pellet_mean, pellet_sem])
    data_head = np.column_stack([head_mean, head_sem])
    data_start = np.column_stack([start_mean, start_sem])
    #data_end = np.column_stack([end_mean, end_sem])
    np.savetxt(f'{PATH}_pellet.csv', data_pellet, fmt=['%.5f', '%.5f'], delimiter=";")
    np.savetxt(f'{PATH}_head.csv', data_head, fmt=['%.5f', '%.5f'], delimiter=";")
    np.savetxt(f'{PATH}_start.csv', data_start, fmt=['%.5f', '%.5f'], delimiter=";")
    #np.savetxt(f'{PATH}_end.csv', data_end, fmt=['%.5f', '%.5f'], delimiter=";")

main()
