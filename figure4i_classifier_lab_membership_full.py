#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Decode in which lab a mouse was trained based on its behavioral metrics during the three sessions
of the full task variant in which the mouse was determined to be ready for ephys.

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

Guido Meijer
June 22, 2020
"""

import numpy as np
from os.path import join
from paper_behavior_functions import institution_map, QUERY
from dj_tools import fit_psychfunc, dj2pandas
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score, confusion_matrix

# Settings
DECODER = 'forest'          # forest, bayes or regression
NUM_SPLITS = 3              # n in n-fold cross validation
ITERATIONS = 2000           # how often to decode
METRICS = ['threshold_l', 'threshold_r', 'bias_l', 'bias_r', 'lapselow_l', 'lapselow_r',
           'lapsehigh_l', 'lapsehigh_r']
METRICS_CONTROL = ['threshold_l', 'threshold_r', 'bias_l', 'bias_r', 'lapselow_l', 'lapselow_r',
                   'lapsehigh_l', 'lapsehigh_r', 'time_zone']


# Decoding function with n-fold cross validation
def decoding(resp, labels, clf, NUM_SPLITS, random_state):
    kf = KFold(n_splits=NUM_SPLITS, shuffle=True, random_state=random_state)
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


# %% query sessions
    
if QUERY is True:
    from paper_behavior_functions import query_sessions_around_criterion
    from ibl_pipeline import reference, subject, behavior
    use_sessions, _ = query_sessions_around_criterion(criterion='ephys',
                                                      days_from_criterion=[2, 0])
    use_sessions = use_sessions & 'task_protocol LIKE "%biased%"'  # only get biased sessions   
    b = (use_sessions * subject.Subject * subject.SubjectLab * reference.Lab
         * behavior.TrialSet.Trial)
    b2 = b.proj('institution_short', 'subject_nickname', 'task_protocol',
                'trial_stim_contrast_left', 'trial_stim_contrast_right', 'trial_response_choice',
                'task_protocol', 'trial_stim_prob_left', 'trial_feedback_type',
                'trial_response_time', 'trial_stim_on_time', 'time_zone')
    bdat = b2.fetch(order_by='institution_short, subject_nickname, session_start_time, trial_id',
                    format='frame').reset_index()
    behav = dj2pandas(bdat)
    behav['institution_code'] = behav.institution_short.map(institution_map()[0])
    
else:
    behav = pd.read_csv('data', 'Fig4.csv')

biased_fits = pd.DataFrame()
for i, nickname in enumerate(behav['subject_nickname'].unique()):
    if np.mod(i+1, 10) == 0:
        print('Processing data of subject %d of %d' % (i+1,
                                                       len(behav['subject_nickname'].unique())))

    # Get lab and timezone
    lab = behav.loc[behav['subject_nickname'] == nickname, 'institution_code'].unique()[0]
    time_zone = behav.loc[behav['subject_nickname'] == nickname, 'time_zone'].unique()[0]
    if (time_zone == 'Europe/Lisbon') or (time_zone == 'Europe/London'):
        time_zone_number = 0
    elif time_zone == 'America/New_York':
        time_zone_number = -5
    elif time_zone == 'America/Los_Angeles':
        time_zone_number = -7

    # Fit psychometric curve
    left_fit = fit_psychfunc(behav[(behav['subject_nickname'] == nickname)
                                   & (behav['probabilityLeft'] == 80)])
    right_fit = fit_psychfunc(behav[(behav['subject_nickname'] == nickname)
                                    & (behav['probabilityLeft'] == 20)])
    fits = pd.DataFrame(data={'threshold_l': left_fit['threshold'],
                              'threshold_r': right_fit['threshold'],
                              'bias_l': left_fit['bias'],
                              'bias_r': right_fit['bias'],
                              'lapselow_l': left_fit['lapselow'],
                              'lapselow_r': right_fit['lapselow'],
                              'lapsehigh_l': left_fit['lapsehigh'],
                              'lapsehigh_r': right_fit['lapsehigh'],
                              'nickname': nickname, 'lab': lab, 'time_zone': time_zone_number})
    biased_fits = biased_fits.append(fits, sort=False)


# %% Do decoding

# Initialize decoders
print('\nDecoding of lab membership..')
if DECODER == 'forest':
    clf = RandomForestClassifier(n_estimators=100, random_state=424242)
elif DECODER == 'bayes':
    clf = GaussianNB()
elif DECODER == 'regression':
    clf = LogisticRegression(solver='liblinear', multi_class='auto')
else:
    raise Exception('DECODER must be forest, bayes or regression')

# Generate random states for each iteration with a fixed seed
np.random.seed(424242)
random_states = np.random.randint(10000, 99999, ITERATIONS)

# Perform decoding of lab membership
result = pd.DataFrame(columns=['original', 'original_shuffled', 'confusion_matrix',
                               'control', 'control_shuffled', 'control_cm'])
decod = biased_fits.copy()
decoding_set = decod[METRICS].values
control_set = decod[METRICS_CONTROL].values
for i in range(ITERATIONS):
    if np.mod(i+1, 100) == 0:
        print('Iteration %d of %d' % (i+1, ITERATIONS))

    # Original dataset
    result.loc[i, 'original'], conf_matrix = decoding(decoding_set, list(decod['lab']),
                                                      clf, NUM_SPLITS, random_states[i])
    result.loc[i, 'confusion_matrix'] = conf_matrix / conf_matrix.sum(axis=1)[:, np.newaxis]
    result.loc[i, 'original_shuffled'] = decoding(decoding_set, list(decod['lab'].sample(frac=1)),
                                                  clf, NUM_SPLITS, random_states[i])[0]

    # Positive control dataset
    result.loc[i, 'control'], conf_matrix = decoding(control_set, list(decod['lab']),
                                                     clf, NUM_SPLITS, random_states[i])
    result.loc[i, 'control_cm'] = conf_matrix / conf_matrix.sum(axis=1)[:, np.newaxis]
    result.loc[i, 'control_shuffled'] = decoding(control_set, list(decod['lab'].sample(frac=1)),
                                                 clf, NUM_SPLITS, random_states[i])[0]
# Save to csv
result.to_pickle(join('classification_results',
                      'classification_results_full_%s.pkl' % DECODER))
