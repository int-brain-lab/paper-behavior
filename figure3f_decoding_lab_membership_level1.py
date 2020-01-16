#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Decode in which lab a mouse was trained based on its behavioral metrics during the three sessions
of level 1 training in which the mouse was determined to be trained.

As a positive control, the time zone in which the mouse was trained is included in the dataset
since the timezone provides geographical information. Decoding is performed using cross-validated
classification. Chance level is determined by shuffling the lab labels and decoding again.

--------------
Parameters
DECODER:            Which decoder to use: 'bayes', 'forest', or 'regression'
NUM_SPLITS:         The N in N-fold cross validation
ITERATIONS:         Number of times to split the dataset in test and train and decode
METRICS:            List of strings indicating which behavioral metrics to include
                    during decoding of lab membership
METRICS_CONTROL:    List of strings indicating which metrics to use for the positive control
FIG_PATH:           String containing a path where to save the output figure

Guido Meijer
16 Jan 2020
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from os.path import join
import seaborn as sns
from datetime import timedelta
from paper_behavior_functions import (query_sessions_around_criterion, seaborn_style, figpath,
                                      institution_map)
from ibl_pipeline import subject, reference
from dj_tools import dj2pandas, fit_psychfunc
from ibl_pipeline import behavior
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score, confusion_matrix

# Parameters
DECODER = 'forest'           # forest, bayes or regression
NUM_SPLITS = 3              # n in n-fold cross validation
ITERATIONS = 2000           # how often to decode
METRICS = ['perf_easy', 'threshold', 'bias']
METRIS_CONTROL = ['perf_easy', 'threshold', 'bias', 'time_zone']
FIG_PATH = figpath()
SAVE_FIG = False


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
    unique_labels, label_counts = np.unique(labels, return_counts=True)
    cm = confusion_matrix(y_true, y_pred, labels=unique_labels)
    return f1, cm


# Query sessions
sessions = query_sessions_around_criterion(criterion='trained', days_from_criterion=[2, 0])[0]
sessions = sessions * subject.Subject * subject.SubjectLab * reference.Lab

# Create dataframe with behavioral metrics of all mice
learned = pd.DataFrame(columns=['mouse', 'lab', 'perf_easy', 'n_trials',
                                'threshold', 'bias', 'reaction_time',
                                'lapse_low', 'lapse_high', 'time_zone', 'UTC'])

for i, nickname in enumerate(np.unique(sessions.fetch('subject_nickname'))):
    if np.mod(i+1, 10) == 0:
        print('Loading data of subject %d of %d' % (i+1, len(
                np.unique(sessions.fetch('subject_nickname')))))

    # Get only the trials of the 50/50 blocks
    trials = (sessions * behavior.TrialSet.Trial
              & 'subject_nickname = "%s"' % nickname).fetch(format='frame')
    trials = trials.reset_index()

    # Fit a psychometric function to these trials and get fit results
    fit_df = dj2pandas(trials)
    fit_result = fit_psychfunc(fit_df)

    # Get RT, performance and number of trials
    reaction_time = trials['rt'].median()*1000
    perf_easy = trials['correct_easy'].mean()*100
    ntrials_perday = trials.groupby('session_uuid').count()['trial_id'].mean()

    # Add results to dataframe
    learned.loc[i, 'mouse'] = nickname
    learned.loc[i, 'lab'] = (sessions & 'subject_nickname = "%s"' % nickname).fetch(
                                                                    'institution_short')[0]
    learned.loc[i, 'perf_easy'] = perf_easy
    learned.loc[i, 'n_trials'] = ntrials_perday
    learned.loc[i, 'threshold'] = fit_result.loc[0, 'threshold']
    learned.loc[i, 'bias'] = fit_result.loc[0, 'bias']
    learned.loc[i, 'reaction_time'] = reaction_time
    learned.loc[i, 'lapse_low'] = fit_result.loc[0, 'lapselow']
    learned.loc[i, 'lapse_high'] = fit_result.loc[0, 'lapsehigh']

    # Get the time zone and the time of training of the last session
    time_zone = (sessions & 'subject_nickname = "%s"' % nickname).fetch('time_zone')[0]
    if (time_zone == 'Europe/Lisbon') or (time_zone == 'Europe/London'):
        time_zone_number = 0
    elif time_zone == 'America/New_York':
        time_zone_number = -5
    elif time_zone == 'America/Los_Angeles':
        time_zone_number = -7
    session_time = (sessions & 'subject_nickname = "%s"' % nickname).fetch(
                                                            'session_start_time')[-1]
    session_time = (session_time - timedelta(hours=time_zone_number)).time()
    learned.loc[i, 'time_zone'] = time_zone_number
    learned.loc[i, 'UTC'] = session_time.hour

# Drop mice with faulty RT
learned = learned[learned['reaction_time'].notnull()]
learned['lab_number'] = learned.lab.map(institution_map()[0])
learned = learned.sort_values('lab_number')

# Initialize decoders
print('\nDecoding of lab membership..')
if DECODER == 'forest':
    clf = RandomForestClassifier(n_estimators=100)
elif DECODER == 'bayes':
    clf = GaussianNB()
elif DECODER == 'regression':
    clf = LogisticRegression(solver='liblinear', multi_class='auto')
else:
    raise Exception('DECODER must be forest, bayes or regression')

# Perform decoding of lab membership
decoding_result = pd.DataFrame(columns=['original', 'original_shuffled', 'confusion_matrix',
                                        'control', 'control_shuffled', 'control_cm'])
decod = learned.copy()
decoding_set = decod[METRICS].values
control_set = decod[METRIS_CONTROL].values
for i in range(ITERATIONS):
    if np.mod(i+1, 100) == 0:
        print('Iteration %d of %d' % (i+1, ITERATIONS))
    # Original dataset
    decoding_result.loc[i, 'original'], conf_matrix = decoding(
            decoding_set, list(decod['lab_number']), clf, NUM_SPLITS)
    decoding_result.loc[i, 'confusion_matrix'] = (conf_matrix
                                                  / conf_matrix.sum(axis=1)[:, np.newaxis])
    decoding_result.loc[i, 'original_shuffled'] = decoding(decoding_set,
                                                           list(
                                                              decod['lab_number'].sample(frac=1)),
                                                           clf, NUM_SPLITS)[0]
    # Positive control dataset
    decoding_result.loc[i, 'control'], conf_matrix = decoding(
            control_set, list(decod['lab_number']), clf, NUM_SPLITS)
    decoding_result.loc[i, 'control_cm'] = (conf_matrix
                                            / conf_matrix.sum(axis=1)[:, np.newaxis])
    decoding_result.loc[i, 'control_shuffled'] = decoding(control_set,
                                                          list(decod['lab_number'].sample(frac=1)),
                                                          clf, NUM_SPLITS)[0]

# Calculate if decoder performs above chance
chance_level = decoding_result['original_shuffled'].mean()
significance = np.percentile(decoding_result['original'], 2.5)
sig_control = np.percentile(decoding_result['control'], 0.001)
if chance_level > significance:
    print('Classification performance not significanlty above chance')
else:
    print('Above chance classification performance!')

# Plot decoding results
f, ax1 = plt.subplots(1, 1, figsize=(4, 4))
sns.violinplot(data=pd.concat([decoding_result['original'], decoding_result['control']], axis=1),
               color=[0.6, 0.6, 0.6], ax=ax1)
ax1.plot([-1, 2], [chance_level, chance_level], 'r--')
ax1.set(ylabel='Decoding performance (F1 score)', xlim=[-0.8, 1.4], ylim=[0, 0.62],
        xticklabels=['Decoding of\nlab membership', 'Positive\ncontrol\n(w. time zone)'])
# ax1.text(0, 0.6, 'n.s.', fontsize=12, ha='center')
# ax1.text(1, 0.6, '***', fontsize=15, ha='center', va='center')
plt.text(0.7, np.mean(decoding_result['original_shuffled'])-0.04, 'Chance level', color='r')
# plt.setp(ax1.xaxis.get_majorticklabels(), rotation=40)
plt.tight_layout(pad=2)
seaborn_style()

if (DECODER == 'forest') & (SAVE_FIG is True):
    plt.savefig(join(FIG_PATH, 'figure3_decoding_%s_level1.pdf' % DECODER), dpi=300)
    plt.savefig(join(FIG_PATH, 'figure3_decoding_%s_level1.png' % DECODER), dpi=300)
elif SAVE_FIG is True:
    plt.savefig(join(FIG_PATH, 'suppfig3_decoding_%s_level1.pdf' % DECODER), dpi=300)
    plt.savefig(join(FIG_PATH, 'suppfig3_decoding_%s_level1.png' % DECODER), dpi=300)

f, ax1 = plt.subplots(1, 1, figsize=(4.25, 4))
sns.heatmap(data=decoding_result['confusion_matrix'].mean())
ax1.plot([0, 7], [0, 7], '--w')
ax1.set(xticklabels=np.arange(1, len(np.unique(list(decod['lab'])))+1),
        yticklabels=np.arange(1, len(np.unique(list(decod['lab'])))+1),
        ylim=[0, len(np.unique(list(decod['lab'])))],
        xlim=[0, len(np.unique(list(decod['lab'])))],
        title='Normalized Confusion Matrix', ylabel='Actual lab', xlabel='Predicted lab')
plt.setp(ax1.xaxis.get_majorticklabels(), rotation=40)
plt.setp(ax1.yaxis.get_majorticklabels(), rotation=40)
plt.gca().invert_yaxis()
plt.tight_layout(pad=2)

if SAVE_FIG is True:
    plt.savefig(join(FIG_PATH, 'suppfig3_confusion_matrix_%s_level1.pdf' % DECODER), dpi=300)
    plt.savefig(join(FIG_PATH, 'suppfig3_confusion_matrix_%s_level1.png' % DECODER), dpi=300)

f, ax1 = plt.subplots(1, 1, figsize=(4.25, 4))
sns.heatmap(data=decoding_result['control_cm'].mean())
ax1.plot([0, 7], [0, 7], '--w')
ax1.set(xticklabels=np.arange(1, len(np.unique(list(decod['lab'])))+1),
        yticklabels=np.arange(1, len(np.unique(list(decod['lab'])))+1),
        title='Normalized Confusion Matrix', ylabel='Actual lab', xlabel='Predicted lab',
        ylim=[0, len(np.unique(list(decod['lab'])))],
        xlim=[0, len(np.unique(list(decod['lab'])))])
plt.setp(ax1.xaxis.get_majorticklabels(), rotation=40)
plt.setp(ax1.yaxis.get_majorticklabels(), rotation=40)
plt.gca().invert_yaxis()
plt.tight_layout(pad=2)

if SAVE_FIG is True:
    plt.savefig(join(FIG_PATH,
                     'suppfig3_control_confusion_matrix_%s_level1.pdf' % DECODER), dpi=300)
    plt.savefig(join(FIG_PATH,
                     'suppfig3_control_confusion_matrix_%s_level1.png' % DECODER), dpi=300)
