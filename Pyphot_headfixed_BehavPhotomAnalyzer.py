import copy
import os
import numpy as np
import scipy.io
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from os.path import join
import seaborn
from analysis.code.tools import data_import
from sklearn.metrics import auc
matplotlib.use('Qt5Agg')

"""This Python script is designed to analyze licking and photometry data from behavioral trials, focusing on trial 
selection. It integrates data from various sources, processes the data to identify cues and licks, and analyzes 
licking behavior and photometric responses."""

PATH_ML = r'C:\Users\stavs\Downloads\Marielle\MM16\230102_MM16_Sucrose-surprise.mat'
PATH_PHOTOMETRY = r'C:\Users\stavs\Downloads\Marielle\MM16\MM16-2023-01-02-122843.ppd'
TITLE1 = 'H2O' # the first condition in the monkeylogic
TITLE2 = 'AceK' # the second condition in the monkeylogic
TITLE3 = 'No Reward' # the third condition in the monkeylogic

start_cue_code = 23
stop_cue_code = 24
start_reward_code = 27
monkey_logic_hz = 1000
photometry_hz = 50
space_between_cues = 4 * photometry_hz  # the minimal time between each cue
first_space = 1 * photometry_hz  # the time before the first cue
pre_trial_length = photometry_hz * 4
valve_1 = 6  # condition in monkeylogic
valve_2 = 7  # condition in monkeylogic
no_outcome = 3  # no outcome or neutral cue


def cues_time(data, n_cues):
    """ Finding the cue timing in the trial """
    cues_timing = []
    for trial in range(1, n_cues + 1):  # matlab index
        behavioral_codes = data[f'Trial{trial}']['BehavioralCodes'][0][0][0][0][1]
        codes_timing = data[f'Trial{trial}']['BehavioralCodes'][0][0][0][0][0]
        start_cue = np.where(behavioral_codes == start_cue_code)[0]
        stop_cue = np.where(behavioral_codes == stop_cue_code)[0]
        if start_reward_code in behavioral_codes:
            start_reward = np.where(behavioral_codes == start_reward_code)[0]
            reward_num = (len((np.where(behavioral_codes == start_reward_code))[0]))
            cues_timing.append(
                (codes_timing[start_cue][0][0], codes_timing[stop_cue][0][0], codes_timing[start_reward][0][0],
                 codes_timing[start_reward][-1][0], reward_num))
        else:
            cues_timing.append((codes_timing[start_cue][0][0], codes_timing[stop_cue][0][0]))

    return cues_timing


def calculate_licks(mat_data, data_type, cues_timing):
    """ Rearrange licks by correct trials (start with cue, finish before next cue)
        Down sampling to photometry hz
        """
    licks = [mat_data[f'Trial{i}'][0][0][10][0][0][11][0][0][data_type] for i in
             range(1, len(cues_timing) + 1)]
    licks_array = []
    for i in range(0, len(licks) - 1):
        timing = cues_timing[i]
        cue_array = licks[i][int(timing[0]):]  # the licks from the start of the cue until the end (not including all the ITI)
        iti_array = licks[i + 1][:int(cues_timing[i + 1][0])]  # from the start of the next trial according to the monkylogic until the next cue.
        licks_array.append(list(cue_array) + list(iti_array))  # concect both in order to creat the real trials.
    reshaped_trials = reduce_hz(licks_array)

    return reshaped_trials


def reduce_hz(plot):
    """ Receiving MonkeyLogic array.
        Down sampling to photometry HZ """
    final_array = []
    reduced_size = int(monkey_logic_hz / photometry_hz)
    for trial in plot:
        reduced_array = []
        for i in range(0, len(trial), reduced_size):
            reduced_data = trial[i: i + reduced_size]
            if sum(reduced_data) > 0:
                reduced_array.append(1)
            else:
                reduced_array.append(0)
        final_array.append(reduced_array)
    return final_array


def identify_cues(signals):
    """ Find cues timing in pyPhotometry data """
    signals_frames = []
    current_state = 0
    space = 0
    for i in range(len(signals)):
        if signals_frames:
            space_cues = space_between_cues
        else:
            space_cues = first_space
        if signals[i] == 1:
            if current_state == 1:
                pass
            elif space > space_cues:
                signals_frames.append(i)
                current_state = 1
                space = 0
            else:
                current_state = 1
        else:
            if current_state == 1:
                space = 0
                current_state = 0
            else:
                space += 1
    return signals_frames

def load_mat_file(path):
    """ Loading MonkeyLogic file.
        Returning cues type, down sampled licks by trial
        """
    mat_data = scipy.io.loadmat(path)
    cues = mat_data['TrialRecord'][0][0][7][0]
    cues_timing = cues_time(mat_data, len(cues))
    licks_raw = calculate_licks(mat_data, 0, cues_timing)
    return cues[:-1], licks_raw, cues_timing[:-1]

def load_photometry_data(path):
    """ Load pyPhotometry file.
        Returning the activity, after subtracting the isosbesric signal and the timing of the cues
        """
    data = data_import.import_ppd(path, low_pass=None, high_pass=None)
    analog1 = data['analog_1']
    analog2 = data['analog_2']
    activity = (analog1 - analog2) + np.median(analog2)

    signals = identify_cues(data['digital_1'])

    return activity, signals, data

def lick_selection(licks_output, trial_array, last_frame, min_trials):
    """ Selecting trials with a lick rate within 1 std from the mean of the first X trials """
    lickrate_list = []
    licklist_condition = [licks_output[i] for i in trial_array]
    for array in licklist_condition:
        roi_licks_output = array[:last_frame]
        total_licks = sum(roi_licks_output)
        lickrate = total_licks / int(len(roi_licks_output)/photometry_hz)
        lickrate_list.append(lickrate)
    lickrate_mean = np.mean(lickrate_list[:min_trials])
    lickrate_std = np.std(lickrate_list[:min_trials])
    low_threshold = lickrate_mean - (1*lickrate_std)
    high_threshold = lickrate_mean + (1*lickrate_std)
    thresholds = [low_threshold, high_threshold]
    lick_selection = [trial for trial in lickrate_list if trial > low_threshold and trial < high_threshold]
    ind_selection_lickrate = [index for index, value in enumerate(lickrate_list) if value in lick_selection]
    ind_selection_trialarray = trial_array[ind_selection_lickrate]
    licks_output_selection = [licks_output[i] for i in ind_selection_trialarray]
    return licks_output_selection, ind_selection_trialarray, lickrate_mean, lickrate_std, thresholds
def prosses_licks(licks_array):
    """ deleting the licks that are not the first lick in a sequence (to avoid double counting) """
    licks_arrange = []
    for i in range(len(licks_array)):
        licks = []
        licks_all = np.asarray(licks_array[i])
        for l in range(len(licks_all)):
            if licks_all[l] == 1 and not licks_all[l - 1] == 1:
                licks.append(1)
            else:
                licks.append(0)
                continue
        licks_arrange.append(licks)
    return licks_arrange
def add_pre_trial_length(licks_array, pre_trial_length):
    """ Adding the pre-trial length to the clean licks array """
    licks_organized = copy.deepcopy(licks_array)
    for i in range(len(licks_organized)):
        pre_cue = licks_organized[i - 1][len(licks_organized[i - 1]) - pre_trial_length:]
        licks_organized[i] = (list(pre_cue) + list(licks_organized[i]))
    licks_organized = pd.DataFrame(licks_organized)
    return licks_organized
def calculate_lick_rates(licks_all_trials, bin_size, duration, photometry_hz):
    """ Calculate the lick rates for each trial """
    num_bins = int(duration / (photometry_hz))
    # Get the number of trials in the data frame
    num_trials = licks_all_trials.shape[0]
    # Create an empty list to store the lick rates for each bin
    lick_rates = []
    # Loop through each trial
    for i in range(num_trials):
        # Get the licks for the current trial
        licks = licks_all_trials.iloc[i, :]
        licks = pd.DataFrame(licks)
        bin_lick_rates = []
        # Loop through each bin
        for j in range(num_bins):
            # Get the start and end frames for the current bin
            start_frame = int(j * photometry_hz)
            end_frame = int((j + 1) * photometry_hz)
            # Count the number of licks in the current bin
            bin_licks = len(np.where(licks.iloc[start_frame:end_frame] == 1)[0])
            # Calculate the lick rate for the current bin
            bin_lick_rate = bin_licks / (bin_size / 1000)  # to be in sec
            bin_lick_rates.append(bin_lick_rate)
        lick_rates.append(bin_lick_rates)
    lick_rates = pd.DataFrame(lick_rates)
    return lick_rates

def calculate_df(photomatry_activity, pre_trial_length, photometry_hz):
    """ Calculate the df/f for each trial """
    total_trials = len(photomatry_activity)
    f0_all = np.zeros((total_trials, 1))
    df_all = []
    for i in range(total_trials):
        pre_trial = trials_activity[i - 1][len(trials_activity[i - 1]) - pre_trial_length:]
        trials_activity[i] = (list(pre_trial) + list(trials_activity[i]))
        trial_activity = np.asarray(trials_activity[i])
        f0_all[i] = np.mean(
            trial_activity[pre_trial_length - (1 * photometry_hz):pre_trial_length])  # 1 sec (50 frames) before the cue
        df_all.append((trial_activity - f0_all[i]) / f0_all[i])
    df_all = df_all[1:]  # trial 0 is not a real trial (it's the time before the first trial) don't take it
    df_all = pd.DataFrame(df_all)
    return df_all

def auc_trials(trials, start_point=0, end_point=0):
    """ Calculate the AUC for each response trial per condition """
    timeperiod = np.arange(start_point, end_point, 1)
    auc_list = []
    for _, row in trials.iterrows():
        roi_array = row[start_point:end_point]
        roi_array_nonan = np.nan_to_num(roi_array, nan=0)
        auc_calc = auc(timeperiod, roi_array_nonan)
        auc_list.append(auc_calc)

    return auc_list
if __name__ == '__main__':
    # Define the folder path
    folder = join(PATH_ML + 'trial_selection')
    # Check if the folder exists, if not, create it
    if not os.path.exists(folder):
        os.makedirs(folder)

    # Open the text file in the specified folder
    log_file_path = join(folder, 'output_log.txt')
    with open(log_file_path, 'w') as log_file:
        conditions_played, licks_output, cues_timing = load_mat_file(PATH_ML)
        activity, cues_frames_photometry, Data = load_photometry_data(PATH_PHOTOMETRY)
        trials_activity = np.split(activity, cues_frames_photometry)
        # plot the digital signal and mark the cues timing on it (for validation)
        plt.plot(Data['digital_1'])
        for index in cues_frames_photometry:
            plt.annotate('*', xy=(index, Data['digital_1'][index]))
        plt.show()
        valve_1_ind = np.where(conditions_played[0:len(conditions_played)] == valve_1)[0]
        valve_2_ind = np.where(conditions_played[0:len(conditions_played)] == valve_2)[0]
        no_outcome_ind = np.where(conditions_played[0:len(conditions_played)] == no_outcome)[0]


        # select trials with threshold of licking behaviour (based on first X trial avg +- std)
        licks_clean = prosses_licks(licks_output)
        time_window = 4 * photometry_hz  # 4 sec that include the cue and the reward time
        min_len = min(len(valve_1_ind), len(valve_2_ind)) # for the surprise sucrose experiment (the shortest one)
        # min_len = round(1/3 * min(len(valve_1_ind), len(valve_2_ind))) # for the longer experiments
        lickselection_valve1, trialselection_valve1, lickrate_mean1, lickrate_std1, lickrate_thresholds1 = lick_selection(licks_clean, valve_1_ind, time_window, min_len)
        lickselection_valve2, trialselection_valve2, lickrate_mean2, lickrate_std2, lickrate_thresholds2 = lick_selection(licks_clean, valve_2_ind, time_window, min_len)

        # Save the outputs to the log file
        print("Lick Rate Mean 1:", lickrate_mean1, file=log_file)
        print("Lick Rate Std 1:", lickrate_std1, file=log_file)
        print("Lick Rate Thresholds 1:", lickrate_thresholds1, file=log_file)

        print("Lick Rate Mean 2:", lickrate_mean2, file=log_file)
        print("Lick Rate Std 2:", lickrate_std2, file=log_file)
        print("Lick Rate Thresholds 2:", lickrate_thresholds2, file=log_file)

        print("Trial Selection Valve 1:", trialselection_valve1, file=log_file)
        print("Trial Selection Valve 2:", trialselection_valve2, file=log_file)
        # Add the pre-trial length to the clean licks array
        trials_licks = add_pre_trial_length(licks_clean, pre_trial_length)

       # Calculate the lick rates for each trial
        bins_size = 1000  # Define the bin size (in milliseconds)
        lim_length = 12 * photometry_hz  # Define the limit for the length of the trials
        all_lick_rates = calculate_lick_rates(trials_licks, bins_size, lim_length, photometry_hz)

        lick_rates_valve1 = all_lick_rates.iloc[trialselection_valve1, :]
        lick_rates_valve2 = all_lick_rates.iloc[trialselection_valve2, :]
        lick_rates_valve1_mean = lick_rates_valve1.mean()
        lick_rates_valve2_mean = lick_rates_valve2.mean()
        lick_rates_valve1_mean.to_csv(join(folder, f'lickrate_{TITLE1}.csv'), sep=';')
        lick_rates_valve2_mean.to_csv(join(folder, f'lickrate_{TITLE2}.csv'), sep=';')

        # now analyze the photometry data
        # calculate the df/f for each trial
        df_trials = calculate_df(trials_activity, pre_trial_length, photometry_hz)
        # Plot heatmaps of the data for each condition (valve 1, valve 2, no outcome) to make sure the data is
        # organized correctly. Also plot the data for all trials.
        figures = [(df_trials.iloc[valve_1_ind, :], 'Valve 1', 'valve1.png'),
                   (df_trials.iloc[valve_2_ind, :], 'Valve 2', 'valve2.png'),
                   (df_trials.iloc[no_outcome_ind, :], 'No outcome', 'no_outcome.png'),
                   (df_trials, 'All trials', 'all_trials.png')]

        # Iterate through the figure list and create and save each figure
        for data, title, filename in figures:
            # Calculate the minimum and maximum values for the color scale of the heatmap
            data_min = data.min().min() - 0.005
            data_max = data.max().max() + 0.01
            # Create the figure
            plt.figure()
            seaborn.heatmap(data, vmin=data_min, vmax=data_max)
            plt.axvline(x=pre_trial_length, color='k', linestyle='--')
            plt.axvline(x=pre_trial_length + 2 * photometry_hz, color='k', linestyle='--')
            plt.title(title)
            plt.savefig(join(folder, filename))
            plt.show()

        # Calculate the mean of selected trials from each condition
        df_trials_mean = df_trials.mean(axis=0)  # mean of all trials
        valve_1_trial_mean = df_trials.iloc[trialselection_valve1, :].mean(axis=0)
        valve_2_trial_mean = df_trials.iloc[trialselection_valve2, :].mean(axis=0)
        condition_3_trial_mean = df_trials.iloc[no_outcome_ind, :].mean(axis=0)
        np.savetxt(join(folder, f'{TITLE1}_trial_mean.csv'), valve_1_trial_mean, fmt=['%.5f'], delimiter=";")
        np.savetxt(join(f'{TITLE2}trial_mean.csv'), valve_2_trial_mean, fmt=['%.5f'], delimiter=";")
        np.savetxt(join(f'{TITLE3}trial_mean.csv'), condition_3_trial_mean, fmt=['%.5f'], delimiter=";")


        # Calculate AUC for each response trial per condition between the start of the cue and the end of the reward
        valve1_auc = auc_trials(df_trials.iloc[trialselection_valve1, :], start_point=pre_trial_length, end_point=pre_trial_length+time_window)
        valve2_auc = auc_trials(df_trials.iloc[trialselection_valve2, :], start_point=pre_trial_length, end_point=pre_trial_length+time_window)
        # Save AUC results to the log file
        print("Valve 1 AUC:", valve1_auc, file=log_file)
        print("Valve 2 AUC:", valve2_auc, file=log_file)

