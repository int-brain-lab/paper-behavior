#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Quantify the variability of the time to trained over labs.

@author: Guido Meijer
16 Jan 2020
"""
from os.path import join

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy import stats
import scikit_posthocs as sp

from paper_behavior_functions import (query_subjects, seaborn_style, institution_map,
                                      group_colors, figpath, EXAMPLE_MOUSE,
                                      FIGURE_HEIGHT, FIGURE_WIDTH, QUERY)
from ibl_pipeline.analyses import behavior as behavior_analysis

# Settings
fig_path = figpath()
seaborn_style()

if QUERY is True:
    # Query sessions
    use_subjects = query_subjects()
    ses = ((use_subjects * behavior_analysis.SessionTrainingStatus * behavior_analysis.PsychResults
            & 'training_status = "in_training" OR training_status = "untrainable"')
           .proj('subject_nickname', 'n_trials_stim', 'institution_short')
           .fetch(format='frame')
           .reset_index())
    ses['n_trials'] = [sum(i) for i in ses['n_trials_stim']]
else:
    ses = pd.read_csv(join('data', 'Fig2d.csv'))
    use_subjects = ses['subject_uuid'].unique()  # For counting the number of subjects

# Construct dataframe
training_time = pd.DataFrame(columns=['sessions'], data=ses.groupby('subject_nickname').size())
training_time['trials'] = ses.groupby('subject_nickname').sum()
training_time['lab'] = ses.groupby('subject_nickname')['institution_short'].apply(list).str[0]

# Change lab name into lab number
training_time['lab_number'] = training_time.lab.map(institution_map()[0])
training_time = training_time.sort_values('lab_number')

# Number of sessions to trained for example mouse
example_training_time = \
    training_time.reset_index()[training_time.reset_index()[
        'subject_nickname'].str.match(EXAMPLE_MOUSE)]['sessions']
# example_training_time = training_time.ix[EXAMPLE_MOUSE]['sessions']

#  statistics
# Test normality
_, normal = stats.normaltest(training_time['sessions'])
if normal < 0.05:
    kruskal = stats.kruskal(*[group['sessions'].values
                              for name, group in training_time.groupby('lab')])
    if kruskal[1] < 0.05:  # Proceed to posthocs
        posthoc = sp.posthoc_dunn(training_time, val_col='sessions',
                                  group_col='lab_number')
else:
    anova = stats.f_oneway(*[group['sessions'].values
                             for name, group in training_time.groupby('lab')])
    if anova[1] < 0.05:
        posthoc = sp.posthoc_tukey(training_time, val_col='sessions',
                                   group_col='lab_number')


# %% PLOT

# Set figure style and color palette
use_palette = [[0.6, 0.6, 0.6]] * len(np.unique(training_time['lab']))
use_palette = use_palette + [[1, 1, 0.2]]
lab_colors = group_colors()

# Add all mice to dataframe seperately for plotting
training_time_no_all = training_time.copy()
training_time_no_all.loc[training_time_no_all.shape[0] + 1, 'lab_number'] = 'All'
training_time_all = training_time.copy()
training_time_all['lab_number'] = 'All'
training_time_all = training_time.append(training_time_all)

# print
print(training_time_all.reset_index().groupby(['lab_number'])['subject_nickname'].nunique())

f, (ax1) = plt.subplots(1, 1, figsize=(FIGURE_WIDTH/3, FIGURE_HEIGHT))
sns.set_palette(lab_colors)
sns.swarmplot(y='sessions', x='lab_number', hue='lab_number', data=training_time_no_all,
              palette=lab_colors, ax=ax1, marker='.')
axbox = sns.boxplot(y='sessions', x='lab_number', data=training_time_all,
                    color='white', showfliers=False, ax=ax1)
axbox.artists[-1].set_edgecolor('black')
for j in range(5 * (len(axbox.artists) - 1), 5 * len(axbox.artists)):
    axbox.lines[j].set_color('black')
ax1.set(ylabel='Days to trained', xlabel='')
ax1.get_legend().set_visible(False)
# [tick.set_color(lab_colors[i]) for i, tick in enumerate(ax1.get_xticklabels())]
plt.setp(ax1.xaxis.get_majorticklabels(), rotation=40)
sns.despine(trim=True)
plt.tight_layout()
plt.savefig(join(fig_path, 'figure2d_time_to_trained.pdf'))
plt.savefig(join(fig_path, 'figure2d_time_to_trained.png'), dpi=300)


# SAME FOR TRIALS TO TRAINED
f, (ax1) = plt.subplots(1, 1, figsize=(FIGURE_WIDTH/5, FIGURE_HEIGHT))
sns.set_palette(lab_colors)
sns.swarmplot(y='trials', x='lab_number', hue='lab_number', data=training_time_no_all,
              palette=lab_colors, ax=ax1, marker='.')
axbox = sns.boxplot(y='trials', x='lab_number', data=training_time_all,
                    color='white', showfliers=False, ax=ax1)
axbox.artists[-1].set_edgecolor('black')
for j in range(5 * (len(axbox.artists) - 1), 5 * len(axbox.artists)):
    axbox.lines[j].set_color('black')
ax1.set(ylabel='Trials to trained', xlabel='')
ax1.get_legend().set_visible(False)
# [tick.set_color(lab_colors[i]) for i, tick in enumerate(ax1.get_xticklabels())]
plt.setp(ax1.xaxis.get_majorticklabels(), rotation=40)
sns.despine(trim=True)
plt.tight_layout()
plt.savefig(join(fig_path, 'figure2d_trials_to_trained.pdf'))
plt.savefig(join(fig_path, 'figure2d_trials_to_trained.png'), dpi=300)


# sns.swarmplot(y='trials', x='lab_number', hue='lab_number', data=training_time_no_all,
#               palette=lab_colors, ax=ax2)
# axbox = sns.boxplot(y='trials', x='lab_number', data=training_time_all,
#                     color='white', showfliers=False, ax=ax2)
# axbox.artists[-1].set_edgecolor('black')
# for j in range(5 * (len(axbox.artists) - 1), 5 * len(axbox.artists)):
#     axbox.lines[j].set_color('black')
# ax2.set(ylabel='Trials to trained', xlabel='', ylim=[0, 50000])
# ax2.get_legend().set_visible(False)
# # [tick.set_color(lab_colors[i]) for i, tick in enumerate(ax1.get_xticklabels())]
# plt.setp(ax2.xaxis.get_majorticklabels(), rotation=40)

# Get stats in text
# Interquartile range per lab
iqtr = training_time.groupby(['lab'])[
    'sessions'].quantile(0.75) - training_time.groupby(['lab'])[
    'sessions'].quantile(0.25)

# Training time as a whole
m_train = training_time['sessions'].mean()
s_train = training_time['sessions'].std()
slowest = training_time['sessions'].max()
fastest = training_time['sessions'].min()

# Print information used in the paper
print('For mice that learned the task, the average training took %.1f ± %.1f days (s.d., '
      'n = %d), similar to the %d days of the example mouse from Lab 1 (Figure 2a, black). The '
      'fastest learners met training criteria in %d days, the slowest %d days'
      % (m_train, s_train, len(use_subjects), example_training_time, fastest, slowest))
