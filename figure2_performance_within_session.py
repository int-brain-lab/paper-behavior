#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 26 13:47:20 2019

Quantify behavioral performance within a session

@author: Guido Meijer
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from os.path import join, expanduser
import seaborn as sns
from paper_behavior_functions import query_sessions, seaborn_style
import datajoint as dj
from ibl_pipeline import subject, acquisition, behavior
from ibl_pipeline.analyses import behavior as behavior_analysis

# Settings
FIG_PATH = join(expanduser('~'), 'Figures', 'Behavior')
TRAINING_PHASE = 'biased'  # biased or training
WINDOW_SIZE = 51  # must be an uneven number
STEP_SIZE = 5

# Query list of sessions
sessions = query_sessions(protocol=TRAINING_PHASE, trained=True, as_dataframe=True)

# Create dataframe
perf_df = pd.DataFrame(columns=['mouse', 'lab', 'perf', 'centers', 'num_trials',
                                'window_size', 'step_size'])

for i, ses_start in enumerate(sessions['session_start_time']):
    if np.mod(i+1, 10) == 0:
        print('Loading data of session %d of %d' % (i+1, len(sessions)))

    # Load in session data
    trials = behavior.TrialSet.Trial & 'session_start_time="%s"' % ses_start
    correct, contr_l, contr_r = trials.fetch('trial_feedback_type', 'trial_stim_contrast_left',
                                             'trial_stim_contrast_right')

    # Calculate percentage correct for high contrast trials with sliding window
    high_contrast = np.abs(contr_l - contr_r) >= 0.5
    centers = np.arange(np.median(np.arange(WINDOW_SIZE)),
                        np.size(correct) - np.median(np.arange(WINDOW_SIZE)), STEP_SIZE, dtype=int)
    perf = np.zeros(np.size(centers))
    num_trials = np.zeros(np.size(centers))
    for j, c in enumerate(centers):
        window = correct[c-int((WINDOW_SIZE-1)/2):c+int((WINDOW_SIZE-1)/2)]
        window = window[high_contrast[c-int((WINDOW_SIZE-1)/2):c+int((WINDOW_SIZE-1)/2)]]
        perf[j] = np.sum(window == 1) / np.size(window)
        num_trials[j] = np.size(window)

    # Add to dataframe
    perf_df.loc[i, 'mouse'] = sessions.loc[i, 'subject_nickname']
    perf_df.loc[i, 'lab'] = sessions.loc[i, 'lab_name']
    perf_df.loc[i, 'perf'] = [perf]
    perf_df.loc[i, 'centers'] = [centers]
    perf_df.loc[i, 'num_trials'] = [num_trials]
    perf_df.loc[i, 'window_size'] = WINDOW_SIZE
    perf_df.loc[i, 'step_size'] = STEP_SIZE


# Function to get average of list of lists with different lengths
def arrays_mean(list_of_arrays):
    arr_lengths = np.array([len(x) for x in list_of_arrays])
    means = np.zeros(max(arr_lengths))
    stderrs = np.zeros(max(arr_lengths))
    for j in range(max(arr_lengths)):
        means[j] = np.mean([el[j] for el in list_of_arrays[arr_lengths > j]])
        stderrs[j] = (np.std([el[j] for el in list_of_arrays[arr_lengths > j]])
                      / np.sqrt(np.sum(arr_lengths > j)))
    return means, stderrs


# Calculate per mouse the average performance of sliding window over sessions
all_perf = []
all_perf_stderr = []
for i, mouse_id in enumerate(np.unique(perf_df['mouse'])):
    arrays = np.array(
            [x for sublist in perf_df.loc[perf_df['mouse'] == mouse_id, 'perf'] for x in sublist])
    this_mean, this_stderr = arrays_mean(arrays)
    all_perf.append(this_mean)
    all_perf_stderr.append(this_stderr)

# Get mean over mice
mean_perf, stderr_perf = arrays_mean(np.array(all_perf))

# Plot results
seaborn_style()
f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(13, 10), sharex=True)
all_centers = np.array([x for sl in perf_df['centers'] for x in sl])
max_centers = all_centers[np.argmax([len(x) for x in all_centers])]

exmpl_ses = 485
sns.lineplot(x=max_centers[0:len(perf_df.loc[exmpl_ses, 'perf'][0])],
             y=perf_df.loc[exmpl_ses, 'perf'][0], color='k', linewidth=2, ax=ax1)
ax1.set(xlim=[0, 1000], ylim=[0.5, 1.01], ylabel='Performance on easy contrasts (%)',
        yticklabels=['50', '60', '70', '80', '90', '100'],
        title='Single example session')

exmpl_mouse = 'IBL-T1'
sns.set_palette(sns.color_palette('Set1'))
for i in perf_df.loc[perf_df['mouse'] == exmpl_mouse, 'perf'].index.values:
    sns.lineplot(x=max_centers[0:len(perf_df.loc[i, 'perf'][0])],
                 y=perf_df.loc[i, 'perf'][0], linewidth=2, ax=ax2)
ax2.set(xlim=[0, 1000], ylim=[0.5, 1.01], ylabel='Performance on easy contrasts (%)',
        yticklabels=['50', '60', '70', '80', '90', '100'],
        title='All sessions for mouse %s' % exmpl_mouse)

exmpl_mouse = 24
sns.lineplot(x=max_centers[0:len(all_perf[exmpl_mouse])], y=all_perf[exmpl_mouse],
             color='k', ax=ax3)
ax3.fill_between(max_centers[0:len(all_perf[exmpl_mouse])],
                 all_perf[exmpl_mouse]-(all_perf_stderr[exmpl_mouse]/2),
                 all_perf[exmpl_mouse]+(all_perf_stderr[exmpl_mouse]/2),
                 alpha=0.5, facecolor=[0.6, 0.6, 0.6])
ax3.set(xlim=[0, 1000], ylim=[0.9, 1.002], ylabel='Performance on easy contrasts (%)',
        yticklabels=np.arange(90, 101, 2),
        xlabel='Center of sliding window (trials)',
        title='Mean over sessions for example mouse')

sns.lineplot(x=max_centers, y=mean_perf, color='k', ax=ax4)
ax4.fill_between(max_centers, mean_perf-(stderr_perf/2), mean_perf+(stderr_perf/2),
                 alpha=0.5, facecolor=[0.6, 0.6, 0.6])
ax4.set(xlim=[0, 1000], ylim=[0.9, 1.002], ylabel='Performance on easy contrasts (%)',
        yticklabels=np.arange(90, 101, 2),
        xlabel='Center of sliding window (trials)',
        title='Mean over all %d mice, window size = %d trials, step size = %d trials' % (
                len(all_perf), perf_df.loc[0, 'window_size'], perf_df.loc[0, 'step_size']))

plt.tight_layout(pad=2)
plt.savefig(join(FIG_PATH, 'figure2_panel_perf_within_session_%s.pdf' % TRAINING_PHASE), dpi=300)
plt.savefig(join(FIG_PATH, 'figure2_panel_perf_within_session_%s.png' % TRAINING_PHASE), dpi=300)
