#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 21 10:30:25 2018

Quantify the variability of behavioral metrics within and between labs of mouse behavior of a
single session. The session is the middle session of the three day streak in which a mouse is
deemed to be trained. This script doesn't perform any analysis but plots summary statistics
over labs.

@author: Guido Meijer
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from os.path import join, expanduser
import seaborn as sns
from paper_behavior_functions import query_subjects, seaborn_style, institution_map
from ibl_pipeline import acquisition, behavior
from ibl_pipeline import subject
from ibl_pipeline.analyses import behavior as behavior_analysis

# Settings
fig_path = join(expanduser('~'), 'Figures', 'Behavior')
csv_path = join(expanduser('~'), 'Data', 'Behavior')

# Query sessions
subjects = query_subjects(as_dataframe=True)

# Create dataframe with behavioral metrics of all mice
training_time = pd.DataFrame(columns=['mouse', 'lab', 'sessions', 'trails'])

for i, nickname in enumerate(subjects['subject_nickname']):
    if np.mod(i+1, 10) == 0:
        print('Loading data of subject %d of %d' % (i+1, len(subjects['subject_nickname'])))

    # Get sessions and trials which are flagged as in_training
    ses_start = (acquisition.Session * subject.Subject * behavior_analysis.SessionTrainingStatus
                 & ('subject_nickname = "%s"' % nickname)
                 & 'training_status = "in_training"').proj('session_start_time')
    trials = (ses_start * acquisition.Session * behavior.TrialSet.Trial)

    # Add to dataframe
    training_time.loc[i, 'mouse'] = nickname
    training_time.loc[i, 'lab'] = subjects.loc[subjects['subject_nickname'] == nickname,
                                               'institution_short'].values[0]
    training_time.loc[i, 'sessions'] = len(ses_start)
    training_time.loc[i, 'trials'] = len(trials)

# Change lab name into lab number
training_time['lab_number'] = training_time.lab.map(institution_map())
training_time = training_time.sort_values('lab_number')

# Add all mice to dataframe seperately for plotting
training_time_all = training_time.copy()
training_time_all['lab_number'] = 'All'
training_time_all = training_time.append(training_time_all)

# Set figure style and color palette
use_palette = [[0.6, 0.6, 0.6]] * len(np.unique(training_time['lab']))
use_palette = use_palette + [[1, 1, 0.2]]
sns.set_palette(use_palette)

# Plot behavioral metrics per lab
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))
sns.set_palette(use_palette)
sns.boxplot(y='sessions', x='lab_number', data=training_time_all, ax=ax1)
ax1.set(ylabel='Training duration (sessions)', xlabel='')
plt.setp(ax1.xaxis.get_majorticklabels(), rotation=40)
sns.boxplot(y='trials', x='lab_number', data=training_time_all, ax=ax2)
ax2.set(ylabel='Training duration (trials)', xlabel='')
plt.setp(ax2.xaxis.get_majorticklabels(), rotation=40)
plt.tight_layout(pad=2)
seaborn_style()

plt.savefig(join(fig_path, 'figure2b_training_time.pdf'), dpi=300)
plt.savefig(join(fig_path, 'figure2b_training_time.png'), dpi=300)
