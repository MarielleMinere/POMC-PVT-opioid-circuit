import numpy as np
import scipy.io
import matplotlib.pyplot as plt
from os.path import join
import plotly.express as px

""" This script is used to analyze the licking behavior of the mice during the task."""

PATH_ML = r'C:\Users\mminere\Desktop\Livneh Lab\Monkeylogic\Day1_Sucrose\Sucrose Conditioning\230102_MM6_Sucrose-surprise.mat'
PATH_PHOTOMETRY = r'C:\Users\mminere\Desktop\Livneh Lab\FP Recordings\Day1\Sucrose conditioning\MM6-2023-01-02-144818.ppd'
output_folder = r'C:\Users\mminere\Desktop\Livneh Lab\Raw Analysis FP+ML\May 2023\Sucrose DR curve'
TITLE1 = 'H2O'
TITLE2 = 'Sucrose'
TITLE3 = 'No Reward'
start_cue_code = 23  # the code for the start of the cue
stop_cue_code = 24  # the code for the end of the cue
start_reward_code = 27  # the code for the start of the reward
stop_reward_code = 18  # the code for the end of the reward
monkey_logic_hz = 1000  # MonkeyLogic sampling rate
final_hz = 50  # down sampling to 50 Hz
pre_trial_length = 4 * final_hz
valve_1 = 6  # TTL4 - Suc1 reward
valve_2 = 8  # TTL5 - Suc20 reward
no_outcome = 3  # TTL6 - no outcome

def cues_time(data, n_cues):
    """ Finding the cue timing in the trial """
    cues_timing = []
    for trial in range(1, n_cues + 1):
        behavioral_codes = data[f'Trial{trial}']['BehavioralCodes'][0][0][0][0][1]
        codes_timing = data[f'Trial{trial}']['BehavioralCodes'][0][0][0][0][0]
        start_cue = np.where(behavioral_codes == start_cue_code)[0]
        stop_cue = np.where(behavioral_codes == stop_cue_code)[0]
        stop_reward = np.where(behavioral_codes == stop_reward_code)[0]
        if start_reward_code in behavioral_codes:
            start_reward = np.where(behavioral_codes == start_reward_code)[0]
            cues_timing.append(
                (codes_timing[start_cue][0][0], codes_timing[stop_cue][0][0], codes_timing[start_reward][0][0],
                 codes_timing[stop_reward][0][0]))
        else:
            cues_timing.append((codes_timing[start_cue][0][0], codes_timing[stop_cue][0][0], codes_timing[stop_reward][0][0]))

    return cues_timing


def calculate_licks(mat_data, data_type, cues_timing):
    """ Rearrange licks by correct trials (start with cue, finish before next cue)
        Down sampling to final hz
        """
    licks = [mat_data[f'Trial{i}'][0][0][10][0][0][11][0][0][data_type] for i in range(1, len(cues_timing) + 1)]
    licks_array = []
    for i in range(0, len(licks) - 1):
        timing = cues_timing[i]
        cue_array = licks[i][
                    int(timing[0]):]  # the licks from the start of the cue until the end (not including all the ITI)
        iti_array = licks[i + 1][:int(
            cues_timing[i + 1][0])]  # from the start of the next trial according to the monkylogic until the next cue.
        licks_array.append(list(cue_array) + list(iti_array))  # concect both in order to creat the real trials.
    reshaped_trials = reduce_hz(licks_array)

    return reshaped_trials


def reduce_hz(plot):
    """ Receiving MonkeyLogic array.
        Down sampling to final HZ """
    final_array = []
    reduced_size = int(monkey_logic_hz / final_hz)
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

def load_mat_file(path):
    """ Loading MonkeyLogic file.
        Returning cues type, down sampled licks by trial
        """
    mat_data = scipy.io.loadmat(path)
    cues = mat_data['TrialRecord'][0][0][7][0]
    cues_timing = cues_time(mat_data, len(cues))
    licks_output = calculate_licks(mat_data, 0, cues_timing)
    return cues[:-1], licks_output, cues_timing[:-1]

def prossesing_licks(licks_data, pre_trial_length):
    """ adding the pre trial length to the licks array and cleaning the data from double licks"""
    max_number_of_frames = max([len(frames) for frames in licks_data])
    licks_organized = np.zeros((len(licks_data), max_number_of_frames + pre_trial_length))
    for i in range(len(licks_data)):
        licks = []
        pre_cue = licks_data[i - 1][len(licks_data[i - 1]) - pre_trial_length:]
        licks_data[i] = (list(pre_cue) + list(licks_data[i]))
        pellet = np.asarray(licks_data[i])
        for l in range(len(pellet)):
            if pellet[l] == 1 and not pellet[l - 1] == 1:
                licks.append(l)
            else:
                continue
        licks_organized[i, :len(licks)] = licks
    licks_organized[licks_organized == 0] = None
    return licks_organized

if '__main__' == __name__:
    conditions_played, licks_all, cues_timing = load_mat_file(PATH_ML)
    valve_1_ind = np.where(conditions_played[0:len(conditions_played)] == valve_1)[0]
    valve_2_ind = np.where(conditions_played[0:len(conditions_played)] == valve_2)[0]
    no_outcome_ind = np.where(conditions_played[0:len(conditions_played)] == no_outcome)[0]

    trials_licks = prossesing_licks(licks_all, pre_trial_length)
    # plot the inter-lick intervals histogram for all trials to make sure the data is clean
    ITI_licks = np.zeros((len(licks_all), np.size(trials_licks, 1) - 1))
    for trial in range(len(trials_licks)):
        ITI = []
        for i in range(np.size(trials_licks, 1) - 1):
            ITI.append(((trials_licks[trial, i + 1] - trials_licks[trial, i]) / final_hz) * 1000)
        ITI_licks[trial, :len(ITI)] = ITI

    ITI_licks = np.hstack(ITI_licks)
    fig = px.histogram(ITI_licks[np.where(ITI_licks < 1000)])
    fig.update_traces(xbins=dict(  # bins used for histogram
        start=0,
        end=np.nanmax(ITI_licks),
        size=20
    ))
    fig.update_layout(title="Inter-lick intervals (ms)")
    fig.show()

   # plot the lick raster for all trials, and for each condition separately
    lim_time = 16 * final_hz
    figures = [(1, trials_licks, 'All_trials.png'), (2, trials_licks[no_outcome_ind, :], str(TITLE3)+'.png'),
               (3, trials_licks[valve_1_ind, :], str(TITLE1)+'.png'),
               (4, trials_licks[valve_2_ind, :], str(TITLE2)+'.png')]

    for fig_num, data, filename in figures:
        plt.figure(fig_num)
        plt.eventplot(data,
                      orientation='horizontal',
                      linelengths=0.8,
                      linewidths=0.5,
                      color=[(0.5, 0.5, 0.8)])
        plt.xlim(0, lim_time)
        plt.xticks(np.arange(0, lim_time, step=10*final_hz), np.arange(0,lim_time, step=10*final_hz))
        plt.axvline(x=pre_trial_length, color='k', linestyle='--')
        plt.axvline(x=pre_trial_length + (2 * final_hz), color='k', linestyle='--')
        plt.ylim(-0.5, data.shape[0] - 0.25)
        plt.yticks(np.arange(0, data.shape[0], step=1), np.arange(1, data.shape[0] + 1, step=1))
        plt.gca().invert_yaxis()
        plt.title(filename)
        plt.savefig(join(output_folder, filename))

