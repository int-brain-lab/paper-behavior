#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 21 10:30:25 2018

Quantify the variability of behavioral metrics within and between labs of mouse behavior.
This script doesn't perform any analysis but plots summary statistics over labs.

@author: Guido Meijer
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from os.path import join, expanduser
import seaborn as sns
from paper_behavior_functions import (query_sessions_around_criterion, seaborn_style,
                                      institution_map, group_colors, figpath)
from dj_tools import dj2pandas, fit_psychfunc
from ibl_pipeline import behavior, subject, reference

# Settings
fig_path = figpath()

# Query sessions
sessions = query_sessions_around_criterion(criterion='trained', days_from_criterion=[2, 0])[0]
sessions = sessions * subject.Subject * subject.SubjectLab * reference.Lab

# Create dataframe with behavioral metrics of all mice
learned = pd.DataFrame(columns=['mouse', 'lab', 'perf_easy', 'n_trials',
                                'threshold', 'bias', 'reaction_time',
                                'lapse_low', 'lapse_high'])

for i, nickname in enumerate(np.unique(sessions.fetch('subject_nickname'))):
    if np.mod(i+1, 10) == 0:
        print('Loading data of subject %d of %d' % (i+1, len(
                np.unique(sessions.fetch('subject_nickname')))))

    # Get the trials of the sessions around criterion
    trials = (sessions * behavior.TrialSet.Trial
              & 'subject_nickname = "%s"' % nickname).fetch(format='frame')
    trials = trials.reset_index()

    # Fit a psychometric function to these trials and get fit results
    fit_df = dj2pandas(trials)
    fit_result = fit_psychfunc(fit_df)

    # Calculate performance on easy trials
    perf_easy = (np.sum(fit_df.loc[fit_df['correct_easy'].notnull(), 'correct_easy'])
                 / np.size(fit_df.loc[fit_df['correct_easy'].notnull(), 'correct_easy'])) * 100

    # Add results to dataframe
    learned.loc[i, 'mouse'] = nickname
    learned.loc[i, 'lab'] = (sessions & 'subject_nickname = "%s"' % nickname).fetch(
                                                                    'institution_short')[0]
    learned.loc[i, 'perf_easy'] = perf_easy
    learned.loc[i, 'n_trials'] = fit_result.loc[0, 'ntrials_perday'][0].mean()
    learned.loc[i, 'threshold'] = fit_result.loc[0, 'threshold']
    learned.loc[i, 'bias'] = fit_result.loc[0, 'bias']
    learned.loc[i, 'reaction_time'] = fit_df['rt'].median()*1000
    learned.loc[i, 'lapse_low'] = fit_result.loc[0, 'lapselow']
    learned.loc[i, 'lapse_high'] = fit_result.loc[0, 'lapsehigh']

# Drop mice with faulty RT
learned = learned[learned['reaction_time'].notnull()]

# Change lab name into lab number
learned['lab_number'] = learned.lab.map(institution_map()[0])
learned = learned.sort_values('lab_number')

# Convert to float
learned['perf_easy'] = learned['perf_easy'].astype(float)
learned['reaction_time'] = learned['reaction_time'].astype(float)
learned['n_trials'] = learned['n_trials'].astype(float)
learned['threshold'] = learned['threshold'].astype(float)
learned['bias'] = learned['bias'].astype(float)
learned['lapse_low'] = learned['lapse_low'].astype(float)

# Add all mice to dataframe seperately for plotting
learned_2 = learned.copy()
learned_2['lab'] = 'All'
learned_2['lab_number'] = 'All'
learned_2 = learned.append(learned_2)

# Z-score data
learned_zs = pd.DataFrame()
learned_zs['lab'] = learned['lab']
learned_zs['lab_number'] = learned['lab_number']
learned_zs['Performance'] = stats.zscore(learned['perf_easy'])
learned_zs['Number of trials'] = stats.zscore(learned['n_trials'])
learned_zs['Threshold'] = stats.zscore(learned['threshold'])
learned_zs['Bias'] = stats.zscore(learned['bias'])
learned_zs['Reaction time'] = stats.zscore(learned['reaction_time'])

# Restructure pandas dataframe for plotting
learned_zs_mean = learned_zs.groupby('lab_number').mean()
learned_zs_new = pd.DataFrame({'zscore': learned_zs_mean['Performance'], 'metric': 'Performance',
                               'lab': learned_zs_mean.index.values})
learned_zs_new = learned_zs_new.append(pd.DataFrame({'zscore': learned_zs_mean['Number of trials'],
                                                     'metric': 'Number of trials',
                                                     'lab': learned_zs_mean.index.values}))
learned_zs_new = learned_zs_new.append(pd.DataFrame({'zscore': learned_zs_mean['Threshold'],
                                                     'metric': 'Threshold',
                                                     'lab': learned_zs_mean.index.values}))
learned_zs_new = learned_zs_new.append(pd.DataFrame({'zscore': learned_zs_mean['Bias'],
                                                     'metric': 'Bias',
                                                     'lab': learned_zs_mean.index.values}))
learned_zs_new = learned_zs_new.append(pd.DataFrame({'zscore': learned_zs_mean['Reaction time'],
                                                     'metric': 'Trial duration',
                                                     'lab': learned_zs_mean.index.values}))

# Set color palette
use_palette = [[0.6, 0.6, 0.6]] * len(np.unique(learned['lab']))
use_palette = use_palette + [[1, 1, 0.2]]
sns.set_palette(use_palette)
lab_colors = group_colors()

# Plot behavioral metrics per lab
f, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(1, 5, figsize=(20, 4))
sns.set_palette(use_palette)

sns.boxplot(y='perf_easy', x='lab_number', data=learned_2, ax=ax1)
ax1.set(ylabel='Performance at easy contrasts (%)', ylim=[70, 101], xlabel='')
[tick.set_color(lab_colors[i]) for i, tick in enumerate(ax1.get_xticklabels()[:-1])]
plt.setp(ax1.xaxis.get_majorticklabels(), rotation=40)

sns.boxplot(y='n_trials', x='lab_number', data=learned_2, ax=ax2)
ax2.set(ylabel='Number of trials', ylim=[0, 2000], xlabel='')
[tick.set_color(lab_colors[i]) for i, tick in enumerate(ax2.get_xticklabels()[:-1])]
plt.setp(ax2.xaxis.get_majorticklabels(), rotation=40)

sns.boxplot(y='threshold', x='lab_number', data=learned_2, ax=ax3)
ax3.set(ylabel='Visual threshold (% contrast)', ylim=[-1, 25], xlabel='')
[tick.set_color(lab_colors[i]) for i, tick in enumerate(ax3.get_xticklabels()[:-1])]
plt.setp(ax3.xaxis.get_majorticklabels(), rotation=40)

sns.boxplot(y='bias', x='lab_number', data=learned_2, ax=ax4)
ax4.set(ylabel='Bias (% contrast)', ylim=[-30, 30], xlabel='')
[tick.set_color(lab_colors[i]) for i, tick in enumerate(ax4.get_xticklabels()[:-1])]
plt.setp(ax4.xaxis.get_majorticklabels(), rotation=40)

sns.boxplot(y='reaction_time', x='lab_number', data=learned_2, ax=ax5)
ax5.set(ylabel='Trial duration (ms)', ylim=[0, 1000], xlabel='')
[tick.set_color(lab_colors[i]) for i, tick in enumerate(ax5.get_xticklabels()[:-1])]
plt.setp(ax5.xaxis.get_majorticklabels(), rotation=40)

plt.tight_layout(pad=2)
seaborn_style()
plt.savefig(join(fig_path, 'figure3b_metrics_per_lab.pdf'), dpi=300)
plt.savefig(join(fig_path, 'figure3b_metrics_per_lab.png'), dpi=300)

f, ax1 = plt.subplots(1, 1, figsize=(4.5, 4.5))
sns.swarmplot(x='metric', y='zscore', data=learned_zs_new, hue='lab', palette=group_colors(),
              size=8, ax=ax1)
ax1.plot([-1, 6], [0, 0], 'r--')
ax1.set(ylim=[-1.5, 1.5], ylabel='Deviation from global average (z-score)', xlabel='')
plt.setp(ax1.xaxis.get_majorticklabels(), rotation=40, ha="right")
# plt.setp(ax6.yaxis.get_majorticklabels(), rotation=40)
# ax1.legend(loc=[0.34, 0.01], prop={'size': 9}, ncol=2).set_title('')
# ax1.legend(loc=[0.01, 0.8], prop={'size': 9}, ncol=3).set_title('')
ax1.get_legend().remove()
ax1.yaxis.set_tick_params(labelbottom=True)

plt.tight_layout(pad=2)
seaborn_style()

plt.savefig(join(fig_path, 'figure3c_deviation.pdf'), dpi=300)
plt.savefig(join(fig_path, 'figure3c_deviation.png'), dpi=300)
