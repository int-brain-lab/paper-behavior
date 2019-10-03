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
import matplotlib
import numpy as np
from scipy import stats
from os.path import join, expanduser
import seaborn as sns
from paper_behavior_functions import query_sessions_around_criterium, seaborn_style
from ibl_pipeline import subject, acquisition, behavior
from ibl_pipeline.analyses import behavior as behavior_analysis
from ibl_pipeline.utils import psychofit as psy

# Settings
fig_path = join(expanduser('~'), 'Figures', 'Behavior')
csv_path = join(expanduser('~'), 'Data', 'Behavior')

# Query sessions
sessions = query_sessions_around_criterium(days_from_trained=[3, 0])

# Create dataframe with behavioral metrics of all mice
learning = pd.DataFrame(columns=['mouse', 'lab', 'learned', 'date_learned', 'training_time',
                                 'perf_easy', 'n_trials', 'threshold', 'bias', 'reaction_time',
                                 'lapse_low', 'lapse_high'])

for i, nickname in enumerate(np.unique(sessions['subject_nickname'])):
    if np.mod(i+1, 10) == 0:
        print('Loading data of subject %d of %d' % (i+1, len(
                np.unique(sessions['subject_nickname']))))

    # Select the three sessions for this mouse
    three_ses = sessions[sessions['subject_nickname'] == nickname]
    three_ses = three_ses.reset_index()
    assert len(three_ses) == 3,'Not three sessions found around criterium'

    # Get the trials of these sessions
    trials = (acquisition.Session * behavior.TrialSet.Trial
              & 'session_start_time="%s" OR session_start_time="%s" OR session_start_time="%s"' % (
                      three_ses.loc[0, 'session_start_time'],
                      three_ses.loc[1, 'session_start_time'],
                      three_ses.loc[2, 'session_start_time'])).fetch(format='frame')






    learning.loc[i, 'learned'] = 'trained'
    learning.loc[i, 'date_learned'] = first_trained_session_date
    learning.loc[i, 'training_time'] = sum(behav.session_date < first_trained_session_date)
    learning.loc[i, 'perf_easy'] = float(
            behav.performance_easy[behav.session_date == first_trained_session_date])*100
    psych['n_trials'] = n_trials = [sum(s) for s in psych.n_trials_stim]
    learning.loc[i, 'n_trials'] = float(
            psych.n_trials[psych.session_date == first_trained_session_date])
    learning.loc[i, 'threshold'] = float(
            psych.threshold[psych.session_date == first_trained_session_date])
    learning.loc[i, 'bias'] = float(
            psych.bias[psych.session_date == first_trained_session_date])
    learning.loc[i, 'lapse_low'] = float(
            psych.lapse_low[psych.session_date == first_trained_session_date])
    learning.loc[i, 'lapse_high'] = float(
            psych.lapse_high[psych.session_date == first_trained_session_date])
    if len(rt) == 0:
        learning.loc[i, 'reaction_time'] = np.nan
    elif sum(rt.session_date == first_trained_session_date) == 0:
        learning.loc[i, 'reaction_time'] = float(
                rt.median_reaction_time[np.argmin(np.array(abs(
                        rt.session_date - first_trained_session_date)))])*1000
    else:
        learning.loc[i, 'reaction_time'] = float(
                rt.median_reaction_time[rt.session_date == first_trained_session_date])*1000

    # Add mouse and lab info to dataframe
    learning.loc[i, 'mouse'] = nickname
    lab_name = subj.fetch1('lab_name')
    learning.loc[i, 'lab'] = lab_name

# Select mice that learned
learned = learning.copy()
learned = learned[learned['learned'] == 'trained']

# Merge some labs
pd.options.mode.chained_assignment = None  # deactivate warning
learned.loc[learned['lab'] == 'zadorlab', 'lab'] = 'churchlandlab'
learned.loc[learned['lab'] == 'hoferlab', 'lab'] = 'mrsicflogellab'

# Rename labs
learned.loc[learned['lab'] == 'angelakilab', 'lab'] = 'NYU'
learned.loc[learned['lab'] == 'churchlandlab', 'lab'] = 'CSHL'
learned.loc[learned['lab'] == 'cortexlab', 'lab'] = 'UCL'
learned.loc[learned['lab'] == 'danlab', 'lab'] = 'Berkeley'
learned.loc[learned['lab'] == 'mainenlab', 'lab'] = 'CCU'
learned.loc[learned['lab'] == 'wittenlab', 'lab'] = 'Princeton'
learned.loc[learned['lab'] == 'mrsicflogellab', 'lab'] = 'SWC'

# Save to CSV file
learning.to_csv(join(csv_path, 'all_mice_data.csv'))
learned.to_csv(join(csv_path, 'learned_mice_data.csv'))

# Add (n = x) to lab names
for i in learned.index.values:
    learned.loc[i, 'lab_n'] = (learned.loc[i, 'lab']
                               + ' (n=' + str(sum(learned['lab'] == learned.loc[i, 'lab'])) + ')')

# Convert to float
learned['training_time'] = learned['training_time'].astype(float)
learned['perf_easy'] = learned['perf_easy'].astype(float)
learned['n_trials'] = learned['n_trials'].astype(float)
learned['threshold'] = learned['threshold'].astype(float)
learned['bias'] = learned['bias'].astype(float)
learned['lapse_low'] = learned['lapse_low'].astype(float)
learned['lapse_high'] = learned['lapse_high'].astype(float)
learned['reaction_time'] = learned['reaction_time'].astype(float)

# Add all mice to dataframe seperately for plotting
learned_2 = learned.copy()
learned_2 = learned_2.drop('training_time', axis=1)
learned_2['lab_n'] = 'All (n=%d)' % len(learned)
learned_2 = learned.append(learned_2)
learned_2 = learned_2.sort_values('lab_n')

# Z-score data
learned_zs = pd.DataFrame()
learned_zs['lab_n'] = learned['lab_n']
learned_zs['lab'] = learned['lab']
learned_zs['Performance'] = stats.zscore(learned['perf_easy'])
learned_zs['Number of trials'] = stats.zscore(learned['n_trials'])
learned_zs['Threshold'] = stats.zscore(learned['threshold'])
learned_zs['Bias'] = stats.zscore(learned['bias'])
learned_zs['Reaction time'] = stats.zscore(learned['reaction_time'])

# Restructure pandas dataframe for plotting
learned_zs_mean = learned_zs.groupby('lab').mean()
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
                                                     'metric': 'Reaction time',
                                                     'lab': learned_zs_mean.index.values}))

# Set figure style and color palette
current_palette = sns.color_palette('Set1')
use_palette = [current_palette[-1]]*len(np.unique(learned['lab']))
all_color = [current_palette[5]]
use_palette = all_color + use_palette
sns.set_palette(use_palette)
seaborn_style()


    # Gather behavioral data for subject
    subj = (subject.Subject * subject.SubjectLab & 'subject_nickname="%s"' % nickname)
    training = pd.DataFrame(behavior_analysis.SessionTrainingStatus * subject.Subject
                            & 'subject_nickname="%s"' % nickname)
    behav = pd.DataFrame(((behavior_analysis.BehavioralSummaryByDate * subject.Subject
                           * subject.SubjectLab
                           & 'subject_nickname="%s"' % nickname)).proj(
                                                    'session_date', 'performance_easy').fetch(
                                                    as_dict=True, order_by='session_date'))
    rt = pd.DataFrame(((behavior_analysis.BehavioralSummaryByDate.ReactionTimeByDate
                        * subject.Subject * subject.SubjectLab
                        & 'subject_nickname="%s"' % nickname)).proj(
                                                    'session_date', 'median_reaction_time').fetch(
                                                    as_dict=True, order_by='session_date'))
    psych = pd.DataFrame(((behavior_analysis.BehavioralSummaryByDate.PsychResults
                           * subject.Subject * subject.SubjectLab
                           & 'subject_nickname="%s"' % nickname)).proj(
                                    'session_date', 'n_trials_stim', 'threshold', 'bias',
                                    'lapse_low', 'lapse_high').fetch(
                                            as_dict=True, order_by='session_date'))

    if len(training) == 0:
        print('No data found for subject %s' % nickname)
        continue


    # Find first session in which mouse is trained
    if (sum(training['training_status'] == 'trained') == 0
            & sum(training['training_status'] == 'over40days') == 0):
        learning.loc[i, 'learned'] = 'in training'
        learning.loc[i, 'training_time'] = len(behav)
    elif (sum(training['training_status'] == 'trained') == 0
          & sum(training['training_status'] == 'over40days') > 0):
        learning.loc[i, 'learned'] = 'over40days'
        learning.loc[i, 'training_time'] = len(behav)
    else:
        first_trained_ind = min(
                training.loc[training['training_status'] == 'trained',
                             'session_start_time'].index.values)
        first_day_ind = first_trained_ind - 1  # Get middle session of 3 day streak
        if training.loc[first_day_ind, 'training_status'] == 'wrong session type run':
            continue
        first_trained_session_datetime = training.loc[first_day_ind, 'session_start_time']
        first_trained_session_date = first_trained_session_datetime.date()

    first_trained_session_date = subjects.loc[i, 'date_trained']
    learning.loc[i, 'learned'] = 'trained'
    learning.loc[i, 'date_learned'] = first_trained_session_date
    learning.loc[i, 'training_time'] = sum(behav.session_date < first_trained_session_date)
    learning.loc[i, 'perf_easy'] = float(
            behav.performance_easy[behav.session_date == first_trained_session_date])*100
    psych['n_trials'] = n_trials = [sum(s) for s in psych.n_trials_stim]
    learning.loc[i, 'n_trials'] = float(
            psych.n_trials[psych.session_date == first_trained_session_date])
    learning.loc[i, 'threshold'] = float(
            psych.threshold[psych.session_date == first_trained_session_date])
    learning.loc[i, 'bias'] = float(
            psych.bias[psych.session_date == first_trained_session_date])
    learning.loc[i, 'lapse_low'] = float(
            psych.lapse_low[psych.session_date == first_trained_session_date])
    learning.loc[i, 'lapse_high'] = float(
            psych.lapse_high[psych.session_date == first_trained_session_date])
    if len(rt) == 0:
        learning.loc[i, 'reaction_time'] = np.nan
    elif sum(rt.session_date == first_trained_session_date) == 0:
        learning.loc[i, 'reaction_time'] = float(
                rt.median_reaction_time[np.argmin(np.array(abs(
                        rt.session_date - first_trained_session_date)))])*1000
    else:
        learning.loc[i, 'reaction_time'] = float(
                rt.median_reaction_time[rt.session_date == first_trained_session_date])*1000

    # Add mouse and lab info to dataframe
    learning.loc[i, 'mouse'] = nickname
    lab_name = subj.fetch1('lab_name')
    learning.loc[i, 'lab'] = lab_name

# Select mice that learned
learned = learning.copy()
learned = learned[learned['learned'] == 'trained']

# Merge some labs
pd.options.mode.chained_assignment = None  # deactivate warning
learned.loc[learned['lab'] == 'zadorlab', 'lab'] = 'churchlandlab'
learned.loc[learned['lab'] == 'hoferlab', 'lab'] = 'mrsicflogellab'

# Rename labs
learned.loc[learned['lab'] == 'angelakilab', 'lab'] = 'NYU'
learned.loc[learned['lab'] == 'churchlandlab', 'lab'] = 'CSHL'
learned.loc[learned['lab'] == 'cortexlab', 'lab'] = 'UCL'
learned.loc[learned['lab'] == 'danlab', 'lab'] = 'Berkeley'
learned.loc[learned['lab'] == 'mainenlab', 'lab'] = 'CCU'
learned.loc[learned['lab'] == 'wittenlab', 'lab'] = 'Princeton'
learned.loc[learned['lab'] == 'mrsicflogellab', 'lab'] = 'SWC'

# Save to CSV file
learning.to_csv(join(csv_path, 'all_mice_data.csv'))
learned.to_csv(join(csv_path, 'learned_mice_data.csv'))

# Add (n = x) to lab names
for i in learned.index.values:
    learned.loc[i, 'lab_n'] = (learned.loc[i, 'lab']
                               + ' (n=' + str(sum(learned['lab'] == learned.loc[i, 'lab'])) + ')')

# Convert to float
learned['training_time'] = learned['training_time'].astype(float)
learned['perf_easy'] = learned['perf_easy'].astype(float)
learned['n_trials'] = learned['n_trials'].astype(float)
learned['threshold'] = learned['threshold'].astype(float)
learned['bias'] = learned['bias'].astype(float)
learned['lapse_low'] = learned['lapse_low'].astype(float)
learned['lapse_high'] = learned['lapse_high'].astype(float)
learned['reaction_time'] = learned['reaction_time'].astype(float)

# Add all mice to dataframe seperately for plotting
learned_2 = learned.copy()
learned_2 = learned_2.drop('training_time', axis=1)
learned_2['lab_n'] = 'All (n=%d)' % len(learned)
learned_2 = learned.append(learned_2)
learned_2 = learned_2.sort_values('lab_n')

# Z-score data
learned_zs = pd.DataFrame()
learned_zs['lab_n'] = learned['lab_n']
learned_zs['lab'] = learned['lab']
learned_zs['Performance'] = stats.zscore(learned['perf_easy'])
learned_zs['Number of trials'] = stats.zscore(learned['n_trials'])
learned_zs['Threshold'] = stats.zscore(learned['threshold'])
learned_zs['Bias'] = stats.zscore(learned['bias'])
learned_zs['Reaction time'] = stats.zscore(learned['reaction_time'])

# Restructure pandas dataframe for plotting
learned_zs_mean = learned_zs.groupby('lab').mean()
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
                                                     'metric': 'Reaction time',
                                                     'lab': learned_zs_mean.index.values}))

# Set figure style and color palette
current_palette = sns.color_palette('Set1')
use_palette = [current_palette[-1]]*len(np.unique(learned['lab']))
all_color = [current_palette[5]]
use_palette = all_color + use_palette
sns.set_palette(use_palette)
seaborn_style()

# Plot behavioral metrics per lab
f, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3, figsize=(13, 10), sharey=True)
sns.set_palette(use_palette)

sns.boxplot(x='perf_easy', y='lab_n', data=learned_2, ax=ax1)
ax1.set(title='Performance at easy contrasts (%)', xlim=[80, 101], ylabel='', xlabel='')
ax1.xaxis.tick_top()
plt.setp(ax1.yaxis.get_majorticklabels(), rotation=40)

# sns.boxplot(x='training_time', y='lab_n', data=learned_2, ax=ax2)
# ax2.set(title='Time to reach trained criterion (sessions)', xlim=[0,60], ylabel='', xlabel='')
# ax2.xaxis.tick_top()
# plt.setp(ax2.yaxis.get_majorticklabels(), rotation=40)

sns.boxplot(x='n_trials', y='lab_n', data=learned_2, ax=ax2)
ax2.set(title='Number of trials', xlim=[0, 2000], ylabel='', xlabel='')
ax2.xaxis.tick_top()
plt.setp(ax2.yaxis.get_majorticklabels(), rotation=40)

sns.boxplot(x='threshold', y='lab_n', data=learned_2, ax=ax3)
ax3.set(title='Visual threshold (% contrast)', xlim=[0, 30], ylabel='', xlabel='')
ax3.xaxis.tick_top()
plt.setp(ax3.yaxis.get_majorticklabels(), rotation=40)

sns.boxplot(x='bias', y='lab_n', data=learned_2, ax=ax4)
ax4.set(title='Bias (% contrast)', xlim=[-40, 40], ylabel='', xlabel='')
ax4.xaxis.tick_top()
plt.setp(ax4.yaxis.get_majorticklabels(), rotation=40)

sns.boxplot(x='reaction_time', y='lab_n', data=learned_2, ax=ax5)
ax5.set(title='Reaction time (ms)', xlim=[0, 1000], ylabel='', xlabel='')
ax5.xaxis.tick_top()
plt.setp(ax5.yaxis.get_majorticklabels(), rotation=40)

ax6.get_shared_y_axes().remove(ax6)
yticker = matplotlib.axis.Ticker()
ax6.yaxis.major = yticker
yloc = matplotlib.ticker.AutoLocator()
yfmt = matplotlib.ticker.ScalarFormatter()
ax6.yaxis.set_major_locator(yloc)
ax6.yaxis.set_major_formatter(yfmt)
sns.set_palette('Paired')
sns.swarmplot(x='metric', y='zscore', data=learned_zs_new, hue='lab', size=8, ax=ax6)
ax6.plot([-1, 6], [0, 0], 'r--')
ax6.set(ylim=[-2.5, 2.5], ylabel='Deviation from global average (z-score)', xlabel='')
plt.setp(ax6.xaxis.get_majorticklabels(), rotation=40, ha="right")
# plt.setp(ax6.yaxis.get_majorticklabels(), rotation=40)
ax6.legend(loc=[0.2, 0.01], prop={'size': 9}, ncol=2).set_title('')
ax6.yaxis.set_tick_params(labelbottom=True)
plt.tight_layout(pad=2)
fig = plt.gcf()
fig.set_size_inches((12, 8), forward=False)

plt.savefig(join(fig_path, 'figure5_metrics_per_lab.pdf'), dpi=300)
plt.savefig(join(fig_path, 'figure5_metrics_per_lab.png'), dpi=300)

# Plot lab deviation from global average
f, ax1 = plt.subplots(1, 1, figsize=(5.5, 6))
sns.set_palette('Paired')
sns.swarmplot(x='metric', y='zscore', data=learned_zs_new, hue='lab', size=8, ax=ax1)
ax1.plot([-1, 6], [0, 0], 'r--')
ax1.set(ylim=[-2.5, 2.5], ylabel='Deviation from global average (z-score)', xlabel='')
plt.setp(ax1.xaxis.get_majorticklabels(), rotation=40, ha="right")
ax1.legend(loc=[0.745, 0.01], prop={'size': 9}).set_title('')

plt.tight_layout(pad=3)
fig = plt.gcf()
fig.set_size_inches((5.5, 6), forward=False)
plt.savefig(join(fig_path, 'figure6_panel_deviation.pdf'), dpi=300)
plt.savefig(join(fig_path, 'figure6_panel_deviation.png'), dpi=300)

# Plot heat map of lab deviation
f, ax1 = plt.subplots(1, 1, figsize=(5.5, 5), sharey=True)
sns.heatmap(data=learned_zs.groupby('lab_n').mean(), vmin=-1, vmax=1,
            cmap=sns.color_palette("coolwarm", 100),
            cbar_kws={"ticks": [-1, -0.5, 0, 0.5, 1]}, ax=ax1)
# cbar_kws={'label':'z-scored mean', "ticks":[-1,-0.5,0,0.5,1]}, ax=ax1)
ax1.set(ylabel='', title='Mean per lab (z-scored over labs)')
plt.setp(ax1.xaxis.get_majorticklabels(), rotation=40, ha="right")

plt.tight_layout(pad=3)
fig = plt.gcf()
fig.set_size_inches((5.5, 5), forward=False)
plt.savefig(join(fig_path, 'figure5_heatmap.pdf'), dpi=300)
