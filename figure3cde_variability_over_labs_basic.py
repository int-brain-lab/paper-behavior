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
seaborn_style()
lab_colors = group_colors()
sns.set_palette(lab_colors)

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

vars = ['n_trials', 'perf_easy',  'threshold', 'bias', 'reaction_time']
ylabels =['Number of trials', 'Performance (%)\non easy trials',
          'Contrast threshold (%)', 'Bias (%)', 'Trial duration (ms)' ]
ylims = [[0, 2000],[70, 100], [0, 30], [-30, 30], [0, 2000]]
for v, ylab, ylim in zip(vars, ylabels, ylims):

    f, ax = plt.subplots(1, 1, figsize=(FIGURE_WIDTH/5, FIGURE_HEIGHT))
    sns.swarmplot(y=v, x='lab_number', data=learned_no_all, hue='lab_number',
                  palette=lab_colors, ax=ax, marker='.')
    axbox = sns.boxplot(y=v, x='lab_number', data=learned_2, color='white',
                        showfliers=False, ax=ax)
    ax.set(ylabel=ylab, ylim=ylim, xlabel='')
    # [tick.set_color(lab_colors[i]) for i, tick in enumerate(ax5.get_xticklabels()[:-1])]
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=60)
    axbox.artists[-1].set_edgecolor('black')
    for j in range(5 * (len(axbox.artists) - 1), 5 * len(axbox.artists)):
        axbox.lines[j].set_color('black')
    ax.get_legend().set_visible(False)

    # statistical annotation
    pvalue = stats_tests.loc[stats_tests['variable'] == v, 'p_value']
    if pvalue.to_numpy()[0] < 0.05:
        ax.annotate(num_star(pvalue.to_numpy()[0]),
                         xy=[0.1, 0.8], xycoords='axes fraction', fontsize=5)

    sns.despine(trim=True)
    plt.tight_layout()
    plt.savefig(join(figpath, 'figure3_metrics_%s.pdf'%v))
    plt.savefig(join(figpath, 'figure3_metrics_%s.pdf'%v), dpi=300)

# %%
# Get stats for text
perf_mean = learned['perf_easy'].mean()
perf_std = learned['perf_easy'].std()
thres_mean = learned['threshold'].mean()
thres_std = learned['threshold'].std()
rt_median = learned['reaction_time'].median()
rt_std = learned['reaction_time'].std()
trials_mean = learned['n_trials'].mean()
trials_std = learned['n_trials'].std()
