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
from paper_behavior_functions import (query_subjects, seaborn_style, institution_map,
                                      group_colors, figpath)
from ibl_pipeline.analyses import behavior as behavior_analysis
from scipy import stats
import scikit_posthocs as sp
from lifelines import KaplanMeierFitter

# Settings
fig_path = figpath()

# Query sessions
use_subjects = query_subjects()
ses = (use_subjects * behavior_analysis.SessionTrainingStatus * behavior_analysis.PsychResults
       & 'training_status = "in_training" OR training_status = "untrainable"').proj(
               'subject_nickname', 'n_trials_stim', 'institution_short').fetch(format='frame')
ses = ses.reset_index()
ses['n_trials'] = [sum(i) for i in ses['n_trials_stim']]

# Construct dataframe
training_time = pd.DataFrame(columns=['sessions'], data=ses.groupby('subject_nickname').size())
training_time['trials'] = ses.groupby('subject_nickname').sum()
training_time['lab'] = ses.groupby('subject_nickname')['institution_short'].apply(list).str[0]

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

# Plot cumulative proportion of trained mice over days
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))
for i, lab in enumerate(np.unique(training_time['lab_number'])):
    y, binEdges = np.histogram(training_time.loc[training_time['lab_number'] == lab, 'sessions'],
                               bins=20)
    y = np.cumsum(y)
    y = y / np.max(y)
    bincenters = 0.5*(binEdges[1:]+binEdges[:-1])
    ax1.plot(bincenters, y, '-o', color=lab_colors[i])
ax1.set(ylabel='Cumulative proportion of trained mice', xlabel='Training day',
        xlim=[0, 60], ylim=[0, 1.02])

# Plot hazard rate survival analysis
kmf = KaplanMeierFitter()
for i, lab in enumerate(np.unique(training_time['lab_number'])):
    kmf.fit(training_time.loc[training_time['lab_number'] == lab, 'sessions'].values)
    prob_trained = 1 - kmf.survival_function_
    ax2.step(prob_trained.index.values, prob_trained.values, color=lab_colors[i], lw=2)
ax2.set(ylabel='Probability of being trained', xlabel='Training day',
        title='Inverse survival function', xlim=[0, 60], ylim=[0, 1.02])

sns.despine(trim=True, offset=5)
plt.tight_layout(pad=2)
seaborn_style()
plt.savefig(join(fig_path, 'figure2c_cumulative_proportion_trained.pdf'), dpi=300)
plt.savefig(join(fig_path, 'figure2c_cumulative_proportion_trained.png'), dpi=300)

# Plot number of sessions to trained per lab
training_time_all = training_time.copy()
training_time_all['lab_number'] = 'All'
training_time_all = training_time.append(training_time_all)

f = plt.figure()
grid = plt.GridSpec(1, 3, wspace=0.4, hspace=0.3)
sns.set_palette(lab_colors)

ax1 = plt.subplot(grid[0, :2])
sns.swarmplot(y='sessions', x='lab_number', hue='lab_number', data=training_time, ax=ax1)
axbox = sns.boxplot(y='sessions', x='lab_number', data=training_time, showfliers=False, ax=ax1)
for patch in axbox.artists:
    r, g, b, a = patch.get_facecolor()
    patch.set_facecolor((r, g, b, 0))
ax1.set(ylabel='Days to trained', xlabel='')
ax1.get_legend().set_visible(False)
# [tick.set_color(lab_colors[i]) for i, tick in enumerate(ax1.get_xticklabels())]
plt.setp(ax1.xaxis.get_majorticklabels(), rotation=40)

ax2 = plt.subplot(grid[0, 2], sharey=ax1)
sns.violinplot(y='sessions', x='lab_number',
               data=training_time_all[training_time_all['lab_number'] == 'All'],
               inner=None, color='gray', ax=ax2)
sns.swarmplot(y='sessions', x='lab_number',
              data=training_time_all[training_time_all['lab_number'] == 'All'],
              color='white', edgecolor='gray', ax=ax2)
ax2.set(ylabel='', xlabel='', ylim=[-1, 60])
ax2.get_yaxis().set_visible(False)
ax2.set_frame_on(False)
plt.setp(ax2.xaxis.get_majorticklabels(), rotation=40)

plt.tight_layout(pad=2)
seaborn_style()

plt.savefig(join(fig_path, 'figure2d_training_time_days.pdf'), dpi=300)
plt.savefig(join(fig_path, 'figure2d_training_time_days.png'), dpi=300)

# Plot number of trials to trained per lab
f = plt.figure()
grid = plt.GridSpec(1, 3, wspace=0.4, hspace=0.3)
sns.set_palette(lab_colors)

ax1 = plt.subplot(grid[0, :2])
sns.swarmplot(y='trials', x='lab_number', hue='lab_number', data=training_time, ax=ax1)
axbox = sns.boxplot(y='trials', x='lab_number', data=training_time, showfliers=False, ax=ax1)
for patch in axbox.artists:
    r, g, b, a = patch.get_facecolor()
    patch.set_facecolor((r, g, b, 0))
ax1.set(ylabel='Trials to trained', xlabel='')
ax1.get_legend().set_visible(False)
# [tick.set_color(lab_colors[i]) for i, tick in enumerate(ax1.get_xticklabels())]
plt.setp(ax1.xaxis.get_majorticklabels(), rotation=40)

ax2 = plt.subplot(grid[0, 2], sharey=ax1)
sns.violinplot(y='trials', x='lab_number',
               data=training_time_all[training_time_all['lab_number'] == 'All'],
               inner=None, color='gray', ax=ax2)
sns.swarmplot(y='trials', x='lab_number',
              data=training_time_all[training_time_all['lab_number'] == 'All'],
              color='white', edgecolor='gray', ax=ax2)
ax2.set(ylabel='', xlabel='', ylim=[-500, 50000])
ax2.get_yaxis().set_visible(False)
ax2.set_frame_on(False)
plt.setp(ax2.xaxis.get_majorticklabels(), rotation=40)

plt.tight_layout(pad=2)
seaborn_style()

plt.savefig(join(fig_path, 'figure2d_training_time_trials.pdf'), dpi=300)
plt.savefig(join(fig_path, 'figure2d_training_time_trials.png'), dpi=300)

# Get stats in text
# Interquartile range per lab
iqtr = training_time.groupby(['lab'])[
    'sessions'].quantile(0.75) - training_time.groupby(['lab'])[
    'sessions'].quantile(0.25)
# Training time as a whole
m_train = training_time['sessions'].mean()
s_train = training_time['sessions'].std()
fastest = training_time['sessions'].max()
slowest = training_time['sessions'].min()
