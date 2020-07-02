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
                                      FIGURE_WIDTH, FIGURE_HEIGHT, QUERY)
from dj_tools import dj2pandas, fit_psychfunc
from ibl_pipeline import behavior, subject, reference
import scikit_posthocs as sp

# Initialize
seaborn_style()
figpath = figpath()
pal = group_colors()
institution_map, col_names = institution_map()
col_names = col_names[:-1]

if QUERY is True:
    # query sessions
    use_sessions, _ = query_sessions_around_criterion(criterion='trained',
                                                      days_from_criterion=[2, 0])
    use_sessions = use_sessions & 'task_protocol LIKE "%training%"'  # only get training sessions

    # restrict by list of dicts with uuids for these sessions
    b = (use_sessions * subject.Subject * subject.SubjectLab * reference.Lab
         * behavior.TrialSet.Trial)

    # reduce the size of the fetch
    b2 = b.proj('institution_short', 'subject_nickname', 'task_protocol', 'session_uuid',
                'trial_stim_contrast_left', 'trial_stim_contrast_right', 'trial_response_choice',
                'task_protocol', 'trial_stim_prob_left', 'trial_feedback_type',
                'trial_response_time', 'trial_stim_on_time')

    # construct pandas dataframe
    bdat = b2.fetch(order_by='institution_short, subject_nickname, session_start_time, trial_id',
                    format='frame').reset_index()
    behav = dj2pandas(bdat)
    behav['institution_code'] = behav.institution_short.map(institution_map)
else:
    behav = pd.read_csv(join('data', 'Fig3.csv'))

# Create dataframe with behavioral metrics of all mice
learned = pd.DataFrame(columns=['mouse', 'lab', 'perf_easy', 'n_trials',
                                'threshold', 'bias', 'reaction_time',
                                'lapse_low', 'lapse_high'])

for i, nickname in enumerate(behav['subject_nickname'].unique()):
    if np.mod(i+1, 10) == 0:
        print('Processing data of subject %d of %d' % (i+1,
                                                       len(behav['subject_nickname'].unique())))

    # Get the trials of the sessions around criterion for this subject
    trials = behav[behav['subject_nickname'] == nickname]
    trials = trials.reset_index()

    # Fit a psychometric function to these trials and get fit results
    fit_result = fit_psychfunc(trials)

    # Get RT, performance and number of trials
    reaction_time = trials['rt'].median()*1000
    perf_easy = trials['correct_easy'].mean()*100
    ntrials_perday = trials.groupby('session_uuid').count()['trial_id'].mean()

    # Add results to dataframe
    learned.loc[i, 'mouse'] = nickname
    learned.loc[i, 'lab'] = trials['institution_short'][0]
    learned.loc[i, 'perf_easy'] = perf_easy
    learned.loc[i, 'n_trials'] = ntrials_perday
    learned.loc[i, 'reaction_time'] = reaction_time
    learned.loc[i, 'threshold'] = fit_result.loc[0, 'threshold']
    learned.loc[i, 'bias'] = fit_result.loc[0, 'bias']
    learned.loc[i, 'lapse_low'] = fit_result.loc[0, 'lapselow']
    learned.loc[i, 'lapse_high'] = fit_result.loc[0, 'lapsehigh']

# Drop mice with faulty RT
learned = learned[learned['reaction_time'].notnull()]

# Change lab name into lab number
learned['lab_number'] = learned.lab.map(institution_map)
learned = learned.sort_values('lab_number')

# Convert to float
learned[['perf_easy', 'reaction_time', 'threshold', 'n_trials',
         'bias', 'lapse_low', 'lapse_high']] = learned[['perf_easy', 'reaction_time',
                                                        'threshold', 'n_trials', 'bias',
                                                        'lapse_low', 'lapse_high']].astype(float)

# Stats
stats_tests = pd.DataFrame(columns=['variable', 'test_type', 'p_value'])
posthoc_tests = {}

for i, var in enumerate(['perf_easy', 'reaction_time', 'n_trials', 'threshold', 'bias']):
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

# Add all mice to dataframe seperately for plotting
learned_no_all = learned.copy()
learned_no_all.loc[learned_no_all.shape[0] + 1, 'lab_number'] = 'All'
learned_2 = learned.copy()
learned_2['lab_number'] = 'All'
learned_2 = learned.append(learned_2)

# %%

# Plot behavioral metrics per lab
f, (ax1, ax2, ax3, ax4, ax5, ax6) = plt.subplots(1, 6,
                                                 figsize=(FIGURE_WIDTH*1.1, FIGURE_HEIGHT))
seaborn_style()
lab_colors = group_colors()
sns.set_palette(lab_colors)

sns.swarmplot(y='perf_easy', x='lab_number', data=learned_no_all, hue='lab_number',
              palette=lab_colors, ax=ax1, marker='.')
axbox = sns.boxplot(y='perf_easy', x='lab_number', data=learned_2, color='white',
                    showfliers=False, ax=ax1)
ax1.set(ylabel='Performance (%)\n on easy trials', ylim=[70, 101], xlabel='')
# [tick.set_color(lab_colors[i]) for i, tick in enumerate(ax1.get_xticklabels()[:-1])]
plt.setp(ax1.xaxis.get_majorticklabels(), rotation=60)
axbox.artists[-1].set_edgecolor('black')
for j in range(5 * (len(axbox.artists) - 1), 5 * len(axbox.artists)):
    axbox.lines[j].set_color('black')
ax1.get_legend().set_visible(False)

sns.swarmplot(y='threshold', x='lab_number', data=learned_no_all, hue='lab_number',
              palette=lab_colors, ax=ax2, marker='.')
axbox = sns.boxplot(y='threshold', x='lab_number', data=learned_2, color='white',
                    showfliers=False, ax=ax2)
ax2.set(ylabel='Visual threshold (% contrast)', ylim=[-1, 30], xlabel='')
# [tick.set_color(lab_colors[i]) for i, tick in enumerate(ax2.get_xticklabels()[:-1])]
plt.setp(ax2.xaxis.get_majorticklabels(), rotation=60)
axbox.artists[-1].set_edgecolor('black')
for j in range(5 * (len(axbox.artists) - 1), 5 * len(axbox.artists)):
    axbox.lines[j].set_color('black')
ax2.get_legend().set_visible(False)

sns.swarmplot(y='bias', x='lab_number', data=learned_no_all, hue='lab_number',
              palette=lab_colors, ax=ax3, marker='.')
axbox = sns.boxplot(y='bias', x='lab_number', data=learned_2, color='white', showfliers=False,
                    ax=ax3)
ax3.set(ylabel='Bias (% contrast)', ylim=[-30, 30], xlabel='')
# [tick.set_color(lab_colors[i]) for i, tick in enumerate(ax3.get_xticklabels()[:-1])]
plt.setp(ax3.xaxis.get_majorticklabels(), rotation=60)
axbox.artists[-1].set_edgecolor('black')
for j in range(5 * (len(axbox.artists) - 1), 5 * len(axbox.artists)):
    axbox.lines[j].set_color('black')
ax3.get_legend().set_visible(False)

sns.swarmplot(y='reaction_time', x='lab_number', data=learned_no_all, hue='lab_number',
              palette=lab_colors, ax=ax4, marker='.')
axbox = sns.boxplot(y='reaction_time', x='lab_number', data=learned_2, color='white',
                    showfliers=False, ax=ax4)
ax4.set(ylabel='Trial duration (ms)', ylim=[100, 10000], xlabel='', yscale='log')
# [tick.set_color(lab_colors[i]) for i, tick in enumerate(ax4.get_xticklabels()[:-1])]
plt.setp(ax4.xaxis.get_majorticklabels(), rotation=60)
axbox.artists[-1].set_edgecolor('black')
for j in range(5 * (len(axbox.artists) - 1), 5 * len(axbox.artists)):
    axbox.lines[j].set_color('black')
ax4.get_legend().set_visible(False)

sns.swarmplot(y='n_trials', x='lab_number', data=learned_no_all, hue='lab_number',
              palette=lab_colors, ax=ax5, marker='.')
axbox = sns.boxplot(y='n_trials', x='lab_number', data=learned_2, color='white',
                    showfliers=False, ax=ax5)
ax5.set(ylabel='Number of trials', ylim=[0, 2000], xlabel='')
# [tick.set_color(lab_colors[i]) for i, tick in enumerate(ax5.get_xticklabels()[:-1])]
plt.setp(ax5.xaxis.get_majorticklabels(), rotation=60)
axbox.artists[-1].set_edgecolor('black')
for j in range(5 * (len(axbox.artists) - 1), 5 * len(axbox.artists)):
    axbox.lines[j].set_color('black')
ax5.get_legend().set_visible(False)

correlation_coef, correlation_p
sns.regplot(x='reaction_time', y='n_trials', data=learned_2, color=[0.6, 0.6, 0.6],
            ci=None, scatter=False, ax=ax6, marker='.')
sns.scatterplot(y='n_trials', x='reaction_time', hue='lab_number', data=learned,
                palette=lab_colors, ax=ax6)
ax6.annotate('Coef =' + ' ' + str(round(correlation_coef, 3)) +
             ' ' + '**** p < 0.0001', xy=[50, 2000], fontsize=5)

ax6.set(ylabel='Number of trials', ylim=[0, 2000])
ax6.set(xlabel='Reaction Time (ms)', xlim=[0, 2000])
ax6.get_legend().remove()

# statistical annotation
for i, var in enumerate(['perf_easy', 'threshold',
                         'bias', 'reaction_time', 'n_trials']):
    def num_star(pvalue):
        if pvalue < 0.05:
            stars = '* p < 0.05'
        if pvalue < 0.01:
            stars = '** p < 0.01'
        if pvalue < 0.001:
            stars = '*** p < 0.001'
        if pvalue < 0.0001:
            stars = '**** p < 0.0001'
        return stars

    pvalue = stats_tests.loc[stats_tests['variable'] == var, 'p_value']
    if pvalue.to_numpy()[0] < 0.05:
        axes = [ax1, ax2, ax3, ax4, ax5, ax6]
        axes[i].annotate(num_star(pvalue.to_numpy()[0]),
                         xy=[0.1, 0.8], xycoords='axes fraction', fontsize=5)

sns.despine(trim=True)
plt.tight_layout(w_pad=-0.1)
plt.savefig(join(figpath, 'figure3c-e_all_metrics_per_lab_level1.pdf'))
plt.savefig(join(figpath, 'figure3c-e_all_metrics_per_lab_level1.png'), dpi=300)

# %%
# Plot behavioral metrics per lab
f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(FIGURE_WIDTH*0.6, FIGURE_HEIGHT))

lab_colors = group_colors()
sns.set_palette(lab_colors)

sns.swarmplot(y='perf_easy', x='lab_number', data=learned_no_all, hue='lab_number',
              palette=lab_colors, ax=ax1, marker='.')
axbox = sns.boxplot(y='perf_easy', x='lab_number', data=learned_2, color='white',
                    showfliers=False, ax=ax1)
ax1.set(ylabel='Performance (%)\n on easy trials', ylim=[70, 101], xlabel='')
# [tick.set_color(lab_colors[i]) for i, tick in enumerate(ax1.get_xticklabels()[:-1])]
plt.setp(ax1.xaxis.get_majorticklabels(), rotation=60)
axbox.artists[-1].set_edgecolor('black')
for j in range(5 * (len(axbox.artists) - 1), 5 * len(axbox.artists)):
    axbox.lines[j].set_color('black')
ax1.get_legend().set_visible(False)

sns.swarmplot(y='threshold', x='lab_number', data=learned_no_all, hue='lab_number',
              palette=lab_colors, ax=ax2, marker='.')
axbox = sns.boxplot(y='threshold', x='lab_number', data=learned_2, color='white',
                    showfliers=False, ax=ax2)
ax2.set(ylabel='Visual threshold (% contrast)', ylim=[-1, 25], xlabel='')
# [tick.set_color(lab_colors[i]) for i, tick in enumerate(ax2.get_xticklabels()[:-1])]
plt.setp(ax2.xaxis.get_majorticklabels(), rotation=60)
axbox.artists[-1].set_edgecolor('black')
for j in range(5 * (len(axbox.artists) - 1), 5 * len(axbox.artists)):
    axbox.lines[j].set_color('black')
ax2.get_legend().set_visible(False)

sns.swarmplot(y='bias', x='lab_number', data=learned_no_all, hue='lab_number',
              palette=lab_colors, ax=ax3, marker='.')
axbox = sns.boxplot(y='bias', x='lab_number', data=learned_2, color='white', showfliers=False,
                    ax=ax3)
ax3.set(ylabel='Bias (% contrast)', ylim=[-30, 30], xlabel='')
# [tick.set_color(lab_colors[i]) for i, tick in enumerate(ax3.get_xticklabels()[:-1])]
plt.setp(ax3.xaxis.get_majorticklabels(), rotation=60)
axbox.artists[-1].set_edgecolor('black')
for j in range(5 * (len(axbox.artists) - 1), 5 * len(axbox.artists)):
    axbox.lines[j].set_color('black')
ax3.get_legend().set_visible(False)


# statistical annotation
for i, var in enumerate(['perf_easy', 'threshold', 'bias']):
    def num_star(pvalue):
        if pvalue < 0.05:
            stars = '* p < 0.05'
        if pvalue < 0.01:
            stars = '** p < 0.01'
        if pvalue < 0.001:
            stars = '*** p < 0.001'
        if pvalue < 0.0001:
            stars = '**** p < 0.0001'
        return stars

    pvalue = stats_tests.loc[stats_tests['variable'] == var, 'p_value']
    if pvalue.to_numpy()[0] < 0.05:
        axes = [ax1, ax2, ax3, ax4, ax5, ax6]
        axes[i].annotate(num_star(pvalue.to_numpy()[0]),
                         xy=[0.1, 0.8], xycoords='axes fraction', fontsize=5)

plt.tight_layout()
sns.despine(trim=True)
plt.savefig(join(figpath, 'figure3c-e_metrics_per_lab_level1.pdf'))
plt.savefig(join(figpath, 'figure3c-e_metrics_per_lab_level1.png'), dpi=300)

# %%

# Get stats in text
perf_mean = learned['perf_easy'].mean()
perf_std = learned['perf_easy'].std()
thres_mean = learned['threshold'].mean()
thres_std = learned['threshold'].std()
rt_median = learned['reaction_time'].median()
rt_std = learned['reaction_time'].std()
trials_mean = learned['n_trials'].mean()
trials_std = learned['n_trials'].std()
