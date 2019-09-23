#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 21 10:30:25 2018

Decode in which lab a mouse was trained based on its behavioral metrics during a single session in
which it is determined that the mouse was trained. The session is the middle session of the three
sessions used to determine if a mouse is trained. As a positive control, the time zone in which
the mouse was trained is included in the dataset since the timezone provides geographical
information. Decoding is performed using cross-validated Random Forest classification. Chance level
is determined by shuffling the lab labels and decoding again.

--------------
Parameters
FIG_PATH:           String containing a path where to save the output figure
NUM_SPLITS:         The N in N-fold cross validation
ITERATIONS:         Number of times to split the dataset in test and train and decode
METRICS:            List of strings indicating which behavioral metrics to include
                    during decoding of lab membership
METRICS_CONTROL:    List of strings indicating which metrics to use for the positive control

@author: Guido Meijer
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from os.path import join, expanduser
import seaborn as sns
from paper_behavior_functions import query_subjects, seaborn_style
from ibl_pipeline import subject, reference
from ibl_pipeline.analyses import behavior as behavior_analysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score

# Parameters
FIG_PATH = join(expanduser('~'), 'Figures', 'Behavior')
NUM_SPLITS = 3        # n in n-fold cross validation
ITERATIONS = 2000     # how often to decode
METRICS = ['perf_easy', 'n_trials', 'threshold', 'bias', 'reaction_time']
METRIS_CONTROL = ['perf_easy', 'n_trials', 'threshold', 'bias', 'reaction_time',
                  'time_zone']


# Decoding function with n-fold cross validation
def decoding(resp, labels, clf, NUM_SPLITS):
    kf = KFold(n_splits=NUM_SPLITS, shuffle=True)
    y_pred = np.array([])
    y_true = np.array([])
    for train_index, test_index in kf.split(resp):
        train_resp = resp[train_index]
        test_resp = resp[test_index]
        clf.fit(train_resp, [labels[j] for j in train_index])
        y_pred = np.append(y_pred, clf.predict(test_resp))
        y_true = np.append(y_true, [labels[j] for j in test_index])
    f1 = f1_score(y_true, y_pred, labels=np.unique(labels), average='micro')
    return f1


# Query list of subjects
subjects = query_subjects()

# Create dataframe with behavioral metrics of all mice
learning = pd.DataFrame(columns=['mouse', 'lab', 'time_zone', 'learned', 'date_learned',
                                 'training_time', 'perf_easy', 'n_trials', 'threshold',
                                 'bias', 'reaction_time', 'lapse_low', 'lapse_high'])
for i, nickname in enumerate(subjects['subject_nickname']):
    if np.mod(i+1, 10) == 0:
        print('Loading data of subject %d of %d' % (i+1, len(subjects)))

    # Gather behavioral data for subject
    subj = subject.Subject * subject.SubjectLab & 'subject_nickname="%s"' % nickname
    training = pd.DataFrame(behavior_analysis.SessionTrainingStatus * subject.Subject
                            & 'subject_nickname="%s"' % nickname)
    behav = pd.DataFrame((behavior_analysis.BehavioralSummaryByDate
                          * subject.Subject * subject.SubjectLab
                          & 'subject_nickname="%s"' % nickname).proj(
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
        first_trained_ind = min(training.loc[training['training_status'] == 'trained',
                                             'session_start_time'].index.values)
        first_day_ind = first_trained_ind - 1  # Get middle session of 3 day streak
        if training.loc[first_day_ind, 'training_status'] == 'wrong session type run':
            continue
        first_trained_session_datetime = training.loc[first_day_ind, 'session_start_time']
        first_trained_session_date = first_trained_session_datetime.date()
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
        if sum(rt.session_date == first_trained_session_date) == 0:
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
    lab_time = reference.Lab * reference.LabLocation & 'lab_name="%s"' % lab_name
    time_zone = lab_time.fetch('time_zone')[0]
    if time_zone == ('Europe/Lisbon' or 'Europe/London'):
        time_zone_number = 0
    elif time_zone == 'America/New_York':
        time_zone_number = -5
    elif time_zone == 'America/Los_Angeles':
        time_zone_number = -7
    learning.loc[i, 'time_zone'] = time_zone_number

# Select mice that learned
learned = learning[learning['learned'] == 'trained']

# Merge some labs
pd.options.mode.chained_assignment = None  # deactivate warning
learned.loc[learned['lab'] == 'zadorlab', 'lab'] = 'churchlandlab'
learned.loc[learned['lab'] == 'hoferlab', 'lab'] = 'mrsicflogellab'

# Add (n = x) to lab names
for i in learned.index.values:
    learned.loc[i, 'lab_n'] = learned.loc[i, 'lab'] + ' (n=' + str(sum(
            learned['lab'] == learned.loc[i, 'lab'])) + ')'

# Initialize decoders
print('\nDecoding of lab membership..')
decod = learned
clf_rf = RandomForestClassifier(n_estimators=100)

# Perform decoding of lab membership
decoding_result = pd.DataFrame(columns=['original', 'original_shuffled',
                                        'control', 'control_shuffled'])
decoding_set = decod[METRICS].values
control_set = decod[METRIS_CONTROL].values
for i in range(ITERATIONS):
    if np.mod(i+1, 100) == 0:
        print('Iteration %d of %d' % (i+1, ITERATIONS))
    # Original dataset
    decoding_result.loc[i, 'original'] = decoding(decoding_set,
                                                  list(decod['lab']), clf_rf, NUM_SPLITS)
    decoding_result.loc[i, 'original_shuffled'] = decoding(decoding_set,
                                                           list(decod['lab'].sample(frac=1)),
                                                           clf_rf, NUM_SPLITS)
    # Positive control dataset
    decoding_result.loc[i, 'control'] = decoding(control_set,
                                                 list(decod['lab']), clf_rf, NUM_SPLITS)
    decoding_result.loc[i, 'control_shuffled'] = decoding(control_set,
                                                          list(decod['lab'].sample(frac=1)),
                                                          clf_rf, NUM_SPLITS)


# Calculate if decoder performs above chance (positive values indicate above chance-level)
sig = np.percentile(decoding_result['original']-np.mean(decoding_result['original_shuffled']), 5)
sig_control = np.percentile(decoding_result['control']
                            - np.mean(decoding_result['control_shuffled']), 5)

# Plot decoding results
seaborn_style()
plt.figure(figsize=(4, 5))
fig = plt.gcf()
ax1 = plt.gca()
sns.violinplot(data=pd.concat([decoding_result['original']-decoding_result['original_shuffled'],
                              decoding_result['control']-decoding_result['control_shuffled']],
                              axis=1), color=[0.6, 0.6, 0.6], ax=ax1)
ax1.plot([-1, 5], [0, 0], 'r--')
ax1.set(ylabel='Decoding performance over chance level\n(F1 score)',
        title='Random forest classifier',
        ylim=[-0.4, 0.8], xlim=[-0.8, 1.8],
        xticklabels=['Decoding of\nlab membership', 'Positive\ncontrol'])
ax1.text(0, 0.68, 'n.s.', fontsize=12, ha='center')
ax1.text(1, 0.68, '***', fontsize=15, ha='center', va='center')
plt.setp(ax1.xaxis.get_majorticklabels(), rotation=60)

plt.tight_layout(pad=2)
plt.savefig(join(FIG_PATH, 'figure5_decoding.pdf'), dpi=300)
plt.savefig(join(FIG_PATH, 'figure5_decoding.png'), dpi=300)
