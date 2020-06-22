#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Quantify the variability of the time to trained over labs.

@author: Guido Meijer
16 Jan 2020
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from os.path import join
import seaborn as sns
from ibl_pipeline import subject, reference, acquisition
from paper_behavior_functions import (seaborn_style, institution_map,
                                      group_colors, figpath, CUTOFF_DATE,
                                      FIGURE_HEIGHT, FIGURE_WIDTH)
from ibl_pipeline.analyses import behavior as behavior_analysis
from lifelines import KaplanMeierFitter

# Settings
fig_path = figpath()
seaborn_style()

# Query all mice
all_mice = (subject.Subject * subject.SubjectLab * reference.Lab
            * subject.SubjectProject() & 'subject_project = "ibl_neuropixel_brainwide_01"')
mice_started_training = (all_mice & (acquisition.Session() & 'task_protocol LIKE "%training%"'))
still_training = all_mice.aggr(behavior_analysis.SessionTrainingStatus,
                               session_start_time='max(session_start_time)') \
                                    * behavior_analysis.SessionTrainingStatus - subject.Death \
                                    & 'training_status = "in_training"' \
                                    & 'session_start_time > "%s"' % CUTOFF_DATE
use_subjects = mice_started_training - still_training

# Get training status and training time in number of sessions and trials
ses = (use_subjects
       * behavior_analysis.SessionTrainingStatus
       * behavior_analysis.PsychResults).proj(
               'subject_nickname', 'training_status', 'n_trials_stim', 'institution_short').fetch(
                                                                   format='frame').reset_index()
ses['n_trials'] = [sum(i) for i in ses['n_trials_stim']]

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
ax1.set(ylabel='Cumulative probability of\nreaching trained criterion', xlabel='Training day',
        xlim=[0, 60], ylim=[0, 1.02])
ax1.set_title('Per lab')

# kmf.fit(training_time['sessions'].values, event_observed=training_time['trained'])
# kmf.plot_cumulative_density(ax=ax2)
# ax2.set(ylabel='Cumulative probability of\nreaching trained criterion', xlabel='Training day',
#         title='All labs', xlim=[0, 60], ylim=[0, 1.02])
# ax2.get_legend().set_visible(False)

sns.despine(trim=True, offset=5)
plt.tight_layout()
seaborn_style()
plt.savefig(join(fig_path, 'figure2c_probability_trained.pdf'))
plt.savefig(join(fig_path, 'figure2c_probability_trained.png'), dpi=300)
