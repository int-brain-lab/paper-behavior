#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 21 10:30:25 2018

Quantify the variability of the time to trained over labs.

@author: Guido Meijer
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from os.path import join
import seaborn as sns
from paper_behavior_functions import (query_subjects, seaborn_style, institution_map,
                                      group_colors, figpath)
from ibl_pipeline import acquisition, behavior
from ibl_pipeline import subject
from ibl_pipeline.analyses import behavior as behavior_analysis
from scipy import stats
import scikit_posthocs as sp

# Settings
fig_path = figpath()

# Query sessions
subjects = query_subjects(as_dataframe=True)

# Create dataframe with behavioral metrics of all mice
training_time = pd.DataFrame(columns=['mouse', 'lab', 'sessions', 'trials'])

for i, nickname in enumerate(subjects['subject_nickname']):
    if np.mod(i+1, 10) == 0:
        print('Loading data of subject %d of %d' % (i+1, len(subjects['subject_nickname'])))

    # Get sessions and trials which are flagged as in_training
    ses_start = (acquisition.Session * subject.Subject * behavior_analysis.SessionTrainingStatus
                 & ('subject_nickname = "%s"' % nickname)
                 & 'training_status = "in_training" OR training_status = "untrainable"').proj(
                         'session_start_time')
    trials = (ses_start * acquisition.Session * behavior.TrialSet.Trial)

    # Add to dataframe
    training_time.loc[i, 'mouse'] = nickname
    training_time.loc[i, 'lab'] = subjects.loc[subjects['subject_nickname'] == nickname,
                                               'institution_short'].values[0]
    training_time.loc[i, 'sessions'] = len(ses_start)
    training_time.loc[i, 'trials'] = len(trials)


# Convert to float
training_time['trials'] = training_time['trials'].astype(float)
training_time['sessions'] = training_time['sessions'].astype(float)

# Change lab name into lab number
training_time['lab_number'] = training_time.lab.map(institution_map()[0])
training_time = training_time.sort_values('lab_number')

#  statistics
# Test normality
_, normal = stats.normaltest(training_time['sessions'])
if normal < 0.05:
    kruskal = stats.kruskal(*[group['sessions'].values
                              for name, group in training_time.groupby('lab')])
    if kruskal[1] < 0.05:  # Proceed to posthocs
        posthoc = sp.posthoc_dunn(training_time, val_col='sessions',
                                  group_col='lab')
else:
    anova = stats.f_oneway(*[group['sessions'].values
                             for name, group in training_time.groupby('lab')])
    if anova[1] < 0.05:
        posthoc = sp.posthoc_tukey(training_time, val_col='sessions',
                                   group_col='lab')

# Add all mice to dataframe seperately for plotting
training_time_all = training_time.copy()
training_time_all['lab_number'] = 'All'
training_time_all = training_time.append(training_time_all)

# Set figure style and color palette
use_palette = [[0.6, 0.6, 0.6]] * len(np.unique(training_time['lab']))
use_palette = use_palette + [[1, 1, 0.2]]
lab_colors = group_colors()

# Plot behavioral metrics per lab
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))
sns.set_palette(use_palette)

sns.boxplot(y='sessions', x='lab_number', data=training_time_all, ax=ax1)
ax1.set(ylabel='Days to trained', xlabel='')
[tick.set_color(lab_colors[i]) for i, tick in enumerate(ax1.get_xticklabels()[:-1])]
plt.setp(ax1.xaxis.get_majorticklabels(), rotation=40)

sns.boxplot(y='trials', x='lab_number', data=training_time_all, ax=ax2)
ax2.set(ylabel='Training duration (trials)', xlabel='')
[tick.set_color(lab_colors[i]) for i, tick in enumerate(ax2.get_xticklabels()[:-1])]
plt.setp(ax2.xaxis.get_majorticklabels(), rotation=40)

plt.tight_layout(pad=2)
seaborn_style()
sns.set_palette(use_palette)

plt.savefig(join(fig_path, 'figure2d_training_time.pdf'), dpi=300)
plt.savefig(join(fig_path, 'figure2d_training_time.png'), dpi=300)

# Plot cumulative proportion of trained mice over days
f, ax1 = plt.subplots(1, 1, figsize=(4, 4))
for i, lab in enumerate(np.unique(training_time['lab'])):
    y, binEdges = np.histogram(training_time.loc[training_time['lab'] == lab, 'sessions'], bins=20)
    y = np.cumsum(y)
    y = y / np.max(y)
    bincenters = 0.5*(binEdges[1:]+binEdges[:-1])
    ax1.plot(bincenters, y, '-o', color=lab_colors[i])

ax1.set(ylabel='Cumulative proportion of trained mice', xlabel='Training day',
        xlim=[0, 60], ylim=[0, 1.01])

plt.tight_layout(pad=2)
seaborn_style()
plt.savefig(join(fig_path, 'figure2c_cumulative_proportion_trained.pdf'), dpi=300)
plt.savefig(join(fig_path, 'figure2c_cumulative_proportion_trained.png'), dpi=300)
