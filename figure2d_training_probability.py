#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Quantify the variability of the time to trained over labs.

@author: Guido Meijer, Miles Wells
16 Jan 2020
"""
from os.path import join

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import seaborn as sns

from ibl_pipeline import subject
from ibl_pipeline.analyses import behavior as behavior_analysis
from paper_behavior_functions import (seaborn_style, institution_map, query_subjects,
                                      group_colors, figpath, load_csv, CUTOFF_DATE,
                                      FIGURE_HEIGHT, FIGURE_WIDTH, QUERY)
from lifelines import KaplanMeierFitter

# Settings
fig_path = figpath()
seaborn_style()

if QUERY is True:
    mice_started_training = query_subjects(criterion=None)
    still_training = (mice_started_training.aggr(behavior_analysis.SessionTrainingStatus,
                                                 session_start_time='max(session_start_time)')
                      * behavior_analysis.SessionTrainingStatus - subject.Death
                      & 'training_status = "in_training"'
                      & 'session_start_time > "%s"' % CUTOFF_DATE)
    use_subjects = mice_started_training - still_training

    # Get training status and training time in number of sessions and trials
    ses = ((use_subjects * behavior_analysis.SessionTrainingStatus * behavior_analysis.PsychResults)
           .proj('subject_nickname', 'training_status', 'n_trials_stim', 'institution_short')
           .fetch(format='frame').reset_index())
    ses['n_trials'] = [sum(i) for i in ses['n_trials_stim']]
    ses = ses.drop('n_trials_stim', axis=1).dropna()
else:
    ses = load_csv('Fig2d.csv').dropna()

# Construct dataframe from query
training_time = pd.DataFrame()
for i, nickname in enumerate(ses['subject_nickname'].unique()):
    training_time.loc[i, 'nickname'] = nickname
    training_time.loc[i, 'lab'] = ses.loc[ses['subject_nickname'] == nickname,
                                          'institution_short'].values[0]
    training_time.loc[i, 'sessions'] = sum((ses['subject_nickname'] == nickname)
                                           & ((ses['training_status'] == 'in_training')
                                              | (ses['training_status'] == 'untrainable')))
    training_time.loc[i, 'trials'] = ses.loc[((ses['subject_nickname'] == nickname)
                                              & (ses['training_status'] == 'in_training')),
                                             'n_trials'].sum()
    training_time.loc[i, 'status'] = ses.loc[ses['subject_nickname'] == nickname,
                                             'training_status'].values[-1]
    training_time.loc[i, 'date'] = ses.loc[ses['subject_nickname'] == nickname,
                                           'session_start_time'].values[-1]

# Transform training status into boolean
training_time['trained'] = np.nan
training_time.loc[((training_time['status'] == 'untrainable')
                   | (training_time['status'] == 'in_training')), 'trained'] = 0
training_time.loc[((training_time['status'] != 'untrainable')
                   & (training_time['status'] != 'in_training')), 'trained'] = 1

# Add lab number
training_time['lab_number'] = training_time.lab.map(institution_map()[0])
training_time = training_time.sort_values('lab_number')

# %% PLOT

# Set figure style and color palette
use_palette = [[0.6, 0.6, 0.6]] * len(np.unique(training_time['lab']))
use_palette = use_palette + [[1, 1, 0.2]]
lab_colors = group_colors()
ylim = [-0.02, 1.02]

# Plot hazard rate survival analysis
f, (ax1) = plt.subplots(1, 1, figsize=(FIGURE_WIDTH/3, FIGURE_HEIGHT))

kmf = KaplanMeierFitter()
for i, lab in enumerate(np.unique(training_time['lab_number'])):
    kmf.fit(training_time.loc[training_time['lab_number'] == lab, 'sessions'].values,
            event_observed=training_time.loc[training_time['lab_number'] == lab, 'trained'])
    ax1.step(kmf.cumulative_density_.index.values, kmf.cumulative_density_.values,
             color=lab_colors[i])
kmf.fit(training_time['sessions'].values, event_observed=training_time['trained'])
ax1.step(kmf.cumulative_density_.index.values, kmf.cumulative_density_.values, color='black')
ax1.set(ylabel='Reached proficiency', xlabel='Training day',
        xlim=[0, 60], ylim=ylim)
ax1.set_title('All labs: %d mice' % training_time['nickname'].nunique())

# kmf.fit(training_time['sessions'].values, event_observed=training_time['trained'])
# kmf.plot_cumulative_density(ax=ax2)
# ax2.set(ylabel='Cumulative probability of\nreaching trained criterion', xlabel='Training day',
#         title='All labs', xlim=[0, 60], ylim=[0, 1.02])
# ax2.get_legend().set_visible(False)

sns.despine(trim=True, offset=5)
plt.tight_layout()
seaborn_style()
plt.savefig(join(fig_path, 'figure2d_probability_trained.pdf'))
plt.savefig(join(fig_path, 'figure2d_probability_trained.png'), dpi=300)

# Plot the same figure as a function of trial number
f, (ax1) = plt.subplots(1, 1, figsize=(FIGURE_WIDTH/3, FIGURE_HEIGHT))

kmf = KaplanMeierFitter()
for i, lab in enumerate(np.unique(training_time['lab_number'])):
    kmf.fit(training_time.loc[training_time['lab_number'] == lab, 'trials'].values,
            event_observed=training_time.loc[training_time['lab_number'] == lab, 'trained'])
    ax1.step(kmf.cumulative_density_.index.values, kmf.cumulative_density_.values,
             color=lab_colors[i])
kmf.fit(training_time['trials'].values, event_observed=training_time['trained'])
ax1.step(kmf.cumulative_density_.index.values, kmf.cumulative_density_.values, color='black')
ax1.set(ylabel='Reached proficiency', xlabel='Trial',
        xlim=[0, 40e3], ylim=ylim)
format_fcn = ticker.FuncFormatter(lambda x, pos: '{:,.0f}'.format(x / 1e3) + 'K')
ax1.xaxis.set_major_formatter(format_fcn)
ax1.set_title('All labs: %d mice' % training_time['nickname'].nunique())

sns.despine(trim=True, offset=5)
plt.tight_layout()
seaborn_style()
plt.savefig(join(fig_path, 'figure2d_probability_trained_trials.pdf'))
plt.savefig(join(fig_path, 'figure2d_probability_trained_trials.png'), dpi=300)
