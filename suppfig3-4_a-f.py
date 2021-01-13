#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Quantify the variability of behavioral metrics within and between labs of mouse behavior.
This script doesn't perform any analysis but plots summary statistics over labs.

Alejandro Pan
06 Jan 2020
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from os.path import join
import seaborn as sns
from paper_behavior_functions import (seaborn_style,
                                      institution_map, group_colors, figpath,
                                      FIGURE_WIDTH, FIGURE_HEIGHT,
                                      fit_psychfunc, num_star,
                                      query_session_around_performance)
import scikit_posthocs as sp
from statsmodels.stats.multitest import multipletests


seaborn_style()
figpath = figpath()
pal = group_colors()
institution_map, col_names = institution_map()
col_names = col_names[:-1]


behav = query_session_around_performance(perform_thres=0.9)
behav['institution_code'] = behav.lab_name.map(institution_map)

# Create dataframe with behavioral metrics of all mice
learned = pd.DataFrame(columns=['mouse', 'institution_short', 'perf_easy', 'n_trials',
                                'threshold', 'bias', 'reaction_time',
                                'lapse_low', 'lapse_high', 'trials_per_minute'])

for i, nickname in enumerate(behav['subject_nickname'].unique()):
    if np.mod(i+1, 10) == 0:
        print('Processing data of subject %d of %d' % (i+1,
                                                       len(behav['subject_nickname'].unique())))

    # Get the trials of the sessions around criterion for this subject (first
    # 90% + next session)
    trials = behav[behav['subject_nickname'] == nickname].reset_index()
    # Exclude sessions with less than 4 contrasts
    trials['contrast_set'] = trials.session_start_time.map(
        trials.groupby(['session_start_time'])['signed_contrast'].unique())
    trials = trials.loc[trials['contrast_set'].str.len()>4]
    if len(trials['session_start_time'].unique())<3:
        continue
    # Fit a psychometric function to these trials and get fit results
    fit_result = fit_psychfunc(trials)

    # Get RT, performance and number of trials
    reaction_time = trials['rt'].median()*1000
    perf_easy = trials['correct_easy'].mean()*100
    ntrials_perday = trials.groupby('session_uuid').count()['trial_id'].mean()


    # average trials/minute to normalise by session length
    trials['session_length'] = (trials.session_end_time - trials.session_start_time).astype('timedelta64[m]')
    total_session_length = trials.groupby('session_uuid')['session_length'].mean().sum()
    total_n_trials = trials['trial_id'].count()

    # Add results to dataframe
    learned.loc[i, 'mouse'] = nickname
    learned.loc[i, 'lab'] = trials['institution_short'].iloc[0]
    learned.loc[i, 'perf_easy'] = perf_easy
    learned.loc[i, 'n_trials'] = ntrials_perday
    learned.loc[i, 'reaction_time'] = reaction_time
    learned.loc[i, 'trials_per_minute'] = total_n_trials / total_session_length
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
float_fields = ['perf_easy', 'reaction_time', 'threshold',
                'n_trials', 'bias', 'lapse_low', 'lapse_high', 'trials_per_minute']
learned[float_fields] = learned[float_fields].astype(float)

# %% Stats
stats_tests = pd.DataFrame(columns=['variable', 'test_type', 'p_value'])
posthoc_tests = {}

for i, var in enumerate(['perf_easy', 'reaction_time', 'n_trials', 'threshold', 'bias', 'trials_per_minute']):
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

    # Test for difference in variance
    _, p_var = stats.levene(*[group[var].values for name, group in learned.groupby('lab_number')])

    posthoc_tests['posthoc_'+str(var)] = posthoc
    stats_tests.loc[i, 'variable'] = var
    stats_tests.loc[i, 'test_type'] = test_type
    stats_tests.loc[i, 'p_value'] = test[1]
    stats_tests.loc[i, 'p_value_variance'] = p_var

# Correct for multiple tests
stats_tests['p_value'] = multipletests(stats_tests['p_value'], method='fdr_bh')[1]
stats_tests['p_value_variance'] = multipletests(stats_tests['p_value_variance'],
                                                method='fdr_bh')[1]

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

# %%
vars = ['n_trials', 'perf_easy',  'threshold', 'bias', 'reaction_time', 'trials_per_minute']
ylabels =['Number of trials', 'Performance (%)\non easy trials',
          'Contrast threshold (%)', 'Bias (%)', 'Trial duration (ms)', 'Trials / minute']
ylims = [[0, 2000],[70, 100], [0, 50], [-30, 30], [0, 2000], [0, 30]]
criteria = [[0, 0],[80, 100], [0, 20], [-10, 10], [0, 0], [0, 0]]
order_x = ['Lab 1','Lab 2','Lab 3','Lab 4','Lab 5','Lab 6','Lab 7','All']
for v, ylab, ylim, crit in zip(vars, ylabels, ylims, criteria):

    f, ax = plt.subplots(1, 1, figsize=(FIGURE_WIDTH/5, FIGURE_HEIGHT))
    sns.swarmplot(y=v, x='lab_number', data=learned_no_all, hue='lab_number',
                  palette=lab_colors, ax=ax, marker='.', order=order_x)
    axbox = sns.boxplot(y=v, x='lab_number', data=learned_2, color='white',
                        showfliers=False, ax=ax, order=order_x)
    ax.set(ylabel=ylab, ylim=ylim, xlabel='')
    ax.axhspan(crit[0], crit[1], facecolor='0.2', alpha=0.2)
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
    plt.savefig(join(figpath, 'supplementaryfigure3_metrics_%s.pdf'%v))
    plt.savefig(join(figpath, 'supplementaryfigure3_metrics_%s.pdf'%v), dpi=300)

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
