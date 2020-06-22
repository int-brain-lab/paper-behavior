#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Quantify the variability of behavioral metrics within and between labs of mouse behavior.
This script doesn't perform any analysis but plots summary statistics over labs.

Guido Meijer
16 Jan 2020
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from os.path import join
import seaborn as sns
from paper_behavior_functions import (query_sessions_around_criterion, seaborn_style,
                                      institution_map, group_colors, figpath,
                                      FIGURE_HEIGHT, FIGURE_WIDTH)
from dj_tools import dj2pandas, fit_psychfunc
from ibl_pipeline import behavior, subject, reference
import scikit_posthocs as sp

# Settings
fig_path = figpath()
seaborn_style()

# Query sessions
sessions = query_sessions_around_criterion(criterion='biased', days_from_criterion=[0, 10])[0]
sessions = (sessions * subject.Subject * subject.SubjectLab * reference.Lab
            & 'task_protocol LIKE "%biased%"')

# Create dataframe with behavioral metrics of all mice
learned = pd.DataFrame(columns=['mouse', 'lab', 'perf_easy', 'threshold', 'bias', 'reaction_time',
                                'lapse_low', 'lapse_high', 'n_trials', 'n_sessions'])

for i, nickname in enumerate(np.unique(sessions.fetch('subject_nickname'))):
    if np.mod(i+1, 10) == 0:
        print('Loading data of subject %d of %d' % (i+1, len(
                np.unique(sessions.fetch('subject_nickname')))))

    # Get only the trials of the 50/50 blocks
    trials = (sessions * behavior.TrialSet.Trial
              & 'subject_nickname = "%s"' % nickname
              & 'trial_stim_prob_left = "0.5"').fetch(format='frame')
    trials = trials.reset_index()

    # Fit a psychometric function to these trials and get fit results
    fit_df = dj2pandas(trials)
    fit_result = fit_psychfunc(fit_df)

    # Get performance, reaction time and number of trials
    reaction_time = trials['rt'].median()*1000
    perf_easy = trials['correct_easy'].mean()*100

    # Get all the trials to get number of trials
    trials = (sessions * behavior.TrialSet.Trial
              & 'subject_nickname = "%s"' % nickname).fetch(format='frame')
    trials = trials.reset_index()
    ntrials_perday = trials.groupby('session_uuid').count()['trial_id'].mean()
    nsessions = trials.groupby('session_uuid').size().shape[0]

    # Add results to dataframe
    learned.loc[i, 'mouse'] = nickname
    learned.loc[i, 'lab'] = (sessions & 'subject_nickname = "%s"' % nickname).fetch(
                                                                    'institution_short')[0]
    learned.loc[i, 'perf_easy'] = perf_easy
    learned.loc[i, 'reaction_time'] = reaction_time
    learned.loc[i, 'threshold'] = fit_result.loc[0, 'threshold']
    learned.loc[i, 'bias'] = fit_result.loc[0, 'bias']
    learned.loc[i, 'lapse_low'] = fit_result.loc[0, 'lapselow']
    learned.loc[i, 'lapse_high'] = fit_result.loc[0, 'lapsehigh']
    learned.loc[i, 'n_trials'] = ntrials_perday
    learned.loc[i, 'n_sessions'] = nsessions

# Drop mice with faulty RT
learned = learned[learned['reaction_time'].notnull()]

# Change lab name into lab number
learned['lab_number'] = learned.lab.map(institution_map()[0])
learned = learned.sort_values('lab_number')

# Convert to float
learned[['perf_easy', 'reaction_time', 'threshold', 'n_sessions',
         'bias', 'lapse_low', 'lapse_high', 'n_trials']] = learned[['perf_easy', 'reaction_time',
                                                                    'threshold', 'n_sessions',
                                                                    'bias', 'lapse_low',
                                                                    'lapse_high',
                                                                    'n_trials']].astype(float)

# Add all mice to dataframe seperately for plotting
learned_no_all = learned.copy()
learned_no_all.loc[learned_no_all.shape[0] + 1, 'lab_number'] = 'All'
learned_2 = learned.copy()
learned_2['lab_number'] = 'All'
learned_2 = learned.append(learned_2)

# Stats
stats_tests = pd.DataFrame(columns=['variable', 'test_type', 'p_value'])
posthoc_tests = {}

for i, var in enumerate(['perf_easy', 'reaction_time', 'threshold', 'bias', 'n_trials']):
    _, normal = stats.normaltest(learned[var])

    if normal < 0.05:
        test_type = 'kruskal'
        test = stats.kruskal(*[group[var].values
                               for name, group in learned.groupby('lab_number')])
        if test[1] < 0.05:  # Proceed to posthocs
            posthoc = sp.posthoc_dunn(learned, val_col=var, group_col='lab_number')
        else:
            posthoc = np.nan
    else:
        test_type = 'anova'
        test = stats.f_oneway(*[group[var].values
                                for name, group in learned.groupby('lab_number')])
        if test[1] < 0.05:
            posthoc = sp.posthoc_tukey(learned, val_col=var, group_col='lab_number')
        else:
            posthoc = np.nan

    posthoc_tests['posthoc_'+str(var)] = posthoc
    stats_tests.loc[i, 'variable'] = var
    stats_tests.loc[i, 'test_type'] = test_type
    stats_tests.loc[i, 'p_value'] = test[1]

if (stats.normaltest(learned['n_trials'])[1] < 0.05 or
        stats.normaltest(learned['reaction_time'])[1] < 0.05):
    test_type = 'spearman'
    correlation_coef, correlation_p = stats.spearmanr(learned['reaction_time'],
                                                      learned['n_trials'])
if (stats.normaltest(learned['n_trials'])[1] > 0.05 and
        stats.normaltest(learned['reaction_time'])[1] > 0.05):
    test_type = 'pearson'
    correlation_coef, correlation_p = stats.pearsonr(learned['reaction_time'],
                                                     learned['n_trials'])

# %% Plot behavioral metrics per lab


def num_star(pvalue):
    if pvalue < 0.05:
        stars = '* p < 0.05'
    if pvalue < 0.01:
        stars = '** p < 0.01'
    if pvalue < 0.001:
        stars = '*** p < 0.001'
    if pvalue < 0.0001:
        stars = '**** p < 0.0001'
    if pvalue > 0.05:
        stars = ''
    return stars


# Create plot
f, (ax1, ax2, ax3, ax4, ax5, ax6) = plt.subplots(1, 6, figsize=(20, 4))

lab_colors = group_colors()
sns.set_palette(lab_colors)

sns.swarmplot(y='perf_easy', x='lab_number', data=learned_no_all, hue='lab_number', ax=ax1)
axbox = sns.boxplot(y='perf_easy', x='lab_number', data=learned_2, color='white',
                    showfliers=False, ax=ax1)
axbox.artists[-1].set_edgecolor('black')
for j in range(5 * (len(axbox.artists) - 1), 5 * len(axbox.artists)):
    axbox.lines[j].set_color('black')
ax1.set(ylabel='Performance at easy contrasts (%)', ylim=[70, 101], xlabel='')
# [tick.set_color(lab_colors[i]) for i, tick in enumerate(ax1.get_xticklabels()[:-1])]
plt.setp(ax1.xaxis.get_majorticklabels(), rotation=40)
ax1.get_legend().set_visible(False)
ax1.text(5, 101, num_star(stats_tests.loc[stats_tests['variable'] == 'perf_easy',
                                          'p_value'].to_numpy()[0]))

sns.swarmplot(y='threshold', x='lab_number', data=learned_no_all, hue='lab_number', ax=ax2)
axbox = sns.boxplot(y='threshold', x='lab_number', data=learned_2, color='white',
                    showfliers=False, ax=ax2)
axbox.artists[-1].set_edgecolor('black')
for j in range(5 * (len(axbox.artists) - 1), 5 * len(axbox.artists)):
    axbox.lines[j].set_color('black')
ax2.set(ylabel='Visual threshold (% contrast)', ylim=[-1, 40], xlabel='')
# [tick.set_color(lab_colors[i]) for i, tick in enumerate(ax2.get_xticklabels()[:-1])]
plt.setp(ax2.xaxis.get_majorticklabels(), rotation=40)
ax2.get_legend().set_visible(False)
ax2.annotate(num_star(stats_tests.loc[stats_tests['variable'] == 'threshold',
                                      'p_value'].to_numpy()[0]), xy=[5, 40])

sns.swarmplot(y='bias', x='lab_number', data=learned_no_all, hue='lab_number', ax=ax3)
axbox = sns.boxplot(y='bias', x='lab_number', data=learned_2, color='white', showfliers=False,
                    ax=ax3)
axbox.artists[-1].set_edgecolor('black')
for j in range(5 * (len(axbox.artists) - 1), 5 * len(axbox.artists)):
    axbox.lines[j].set_color('black')
# [tick.set_color(lab_colors[i]) for i, tick in enumerate(ax3.get_xticklabels()[:-1])]
plt.setp(ax3.xaxis.get_majorticklabels(), rotation=40)
ax3.set(ylabel='Bias', ylim=[-30, 30], xlabel='')
ax3.get_legend().set_visible(False)
ax3.annotate(num_star(stats_tests.loc[stats_tests['variable'] == 'bias',
                                      'p_value'].to_numpy()[0]), xy=[5, 5])

sns.swarmplot(y='reaction_time', x='lab_number', data=learned_no_all, hue='lab_number', ax=ax4)
axbox = sns.boxplot(y='reaction_time', x='lab_number', data=learned_2, color='white',
                    showfliers=False, ax=ax4)
axbox.artists[-1].set_edgecolor('black')
for j in range(5 * (len(axbox.artists) - 1), 5 * len(axbox.artists)):
    axbox.lines[j].set_color('black')
ax4.get_legend().set_visible(False)
ax4.set(ylabel='Trial duration (ms)', ylim=[100, 10000], xlabel='', yscale='log')
# [tick.set_color(lab_colors[i]) for i, tick in enumerate(ax4.get_xticklabels()[:-1])]
plt.setp(ax4.xaxis.get_majorticklabels(), rotation=40)
ax4.annotate(num_star(stats_tests.loc[stats_tests['variable'] == 'reaction_time',
                                      'p_value'].to_numpy()[0]), xy=[5, 3500])

sns.swarmplot(y='n_trials', x='lab_number', data=learned_no_all, hue='lab_number', ax=ax5)
axbox = sns.boxplot(y='n_trials', x='lab_number', data=learned_2, color='white',
                    showfliers=False, ax=ax5)
axbox.artists[-1].set_edgecolor('black')
for j in range(5 * (len(axbox.artists) - 1), 5 * len(axbox.artists)):
    axbox.lines[j].set_color('black')
ax5.get_legend().set_visible(False)
ax5.set(ylabel='Number of trials', ylim=[0, 2000], xlabel='')
# [tick.set_color(lab_colors[i]) for i, tick in enumerate(ax5.get_xticklabels()[:-1])]
plt.setp(ax5.xaxis.get_majorticklabels(), rotation=40)
ax5.annotate(num_star(stats_tests.loc[stats_tests['variable'] == 'n_trials',
                                      'p_value'].to_numpy()[0]), xy=[5, 1200])

correlation_coef, correlation_p
sns.regplot(x='reaction_time', y='n_trials', data=learned_2, color=[0.6, 0.6, 0.6],
            ci=None, scatter=False, ax=ax6)
sns.scatterplot(y='n_trials', x='reaction_time', hue='lab_number', data=learned,
                palette=lab_colors, ax=ax6)
ax6.annotate('Coef =' + ' ' + str(round(correlation_coef, 3)) +
             ' ' + '**** p < 0.0001', xy=[300, 2000])

ax6.set(ylabel='Number of trials', ylim=[0, 2000])
ax6.set(xlabel='Reaction Time (ms)', xlim=[0, 2000])
ax6.get_legend().remove()

# statistical annotation
plt.tight_layout(pad=2)
seaborn_style()

plt.savefig(join(fig_path, 'figure3g-i_metrics_per_lab_level2.pdf'), dpi=300)
plt.savefig(join(fig_path, 'figure3g-i_metrics_per_lab_level2.png'), dpi=300)
