#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Decode in which lab a mouse was trained based on its behavioral metrics during the three sessions
of the basic task variant in which the mouse was determined to be trained.

As a positive control, the time zone in which the mouse was trained is included in the dataset
since the timezone provides geographical information. Decoding is performed using leave-one-out
cross-validation. To control for the imbalance in the dataset (some labs have more mice than
others) a fixed number of mice is randomly sub-sampled from each lab. This random sampling is
repeated for a large number of repetitions. A shuffled nul-distribution is obtained by shuffling
the lab labels and decoding again for each iteration.

--------------
Parameters
DECODER:            Which decoder to use: 'bayes', 'forest', or 'regression'
N_MICE:             How many mice per lab to randomly sub-sample
                    (must be lower than the lab with the least mice)
ITERATIONS:         Number of times to randomly sub-sample
METRICS:            List of strings indicating which behavioral metrics to include
                    during decoding of lab membership
METRICS_CONTROL:    List of strings indicating which metrics to use for the positive control

Guido Meijer
September 3, 2020
"""

import pandas as pd
import numpy as np
from os.path import join
from paper_behavior_functions import (query_sessions_around_criterion, institution_map, QUERY,
                                        dj2pandas, fit_psychfunc, datapath)
from ibl_pipeline import subject, reference
from ibl_pipeline import behavior
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import f1_score, confusion_matrix

# Parameters
DECODER = 'bayes'           # bayes, forest or regression
N_MICE = 8                  # how many mice per lab to sub-sample
ITERATIONS = 2000           # how often to decode
METRICS = ['perf_easy', 'threshold', 'bias']
METRICS_CONTROL = ['perf_easy', 'threshold', 'bias', 'time_zone']


# Decoding function with n-fold cross validation
def decoding(data, labels, clf):
    kf = LeaveOneOut()
    y_pred = np.empty(len(labels), dtype='<U5')
    for train_index, test_index in kf.split(data):
        clf.fit(data[train_index], labels[train_index])
        y_pred[test_index] = clf.predict(data[test_index])
    f1 = f1_score(labels, y_pred, average='micro')
    cm = confusion_matrix(labels, y_pred)
    return f1, cm


# %% Query sessions

if QUERY is True:
    use_sessions, _ = query_sessions_around_criterion(criterion='trained',
                                                      days_from_criterion=[2, 0])
    behav = dj2pandas(
        ((use_sessions & 'task_protocol LIKE "%training%"')  # only get training sessions
         * subject.Subject * subject.SubjectLab * reference.Lab * behavior.TrialSet.Trial)

        # Query only the fields we require
        .proj('institution_short', 'subject_nickname', 'task_protocol',
              'trial_stim_contrast_left', 'trial_stim_contrast_right', 'trial_response_choice',
              'task_protocol', 'trial_stim_prob_left', 'trial_feedback_type',
              'trial_response_time', 'trial_stim_on_time', 'time_zone')

        # Fetch as a pandas DataFrame, ordered by institute
        .fetch(order_by='institution_short, subject_nickname, session_start_time, trial_id',
               format='frame')
        .reset_index()
    )
    behav['institution_code'] = behav.institution_short.map(institution_map()[0])
else:
    behav = pd.read_csv(datapath(), 'Fig3.csv')

# Create dataframe with behavioral metrics of all mice
learned = pd.DataFrame(columns=['mouse', 'lab', 'perf_easy', 'n_trials',
                                'threshold', 'bias', 'reaction_time',
                                'lapse_low', 'lapse_high', 'time_zone', 'UTC'])

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
    ntrials_perday = trials.groupby('session_start_time').count()['trial_id'].mean()

     # Get timezone
    time_zone = trials['time_zone'][0]
    if (time_zone == 'Europe/Lisbon') or (time_zone == 'Europe/London'):
        time_zone_number = 0
    elif time_zone == 'America/New_York':
        time_zone_number = -5
    elif time_zone == 'America/Los_Angeles':
        time_zone_number = -7

    # Add results to dataframe
    learned.loc[i, 'mouse'] = nickname
    learned.loc[i, 'lab'] = trials['institution_short'][0]
    learned.loc[i, 'perf_easy'] = perf_easy
    learned.loc[i, 'n_trials'] = ntrials_perday
    learned.loc[i, 'threshold'] = fit_result.loc[0, 'threshold']
    learned.loc[i, 'bias'] = fit_result.loc[0, 'bias']
    learned.loc[i, 'reaction_time'] = reaction_time
    learned.loc[i, 'lapse_low'] = fit_result.loc[0, 'lapselow']
    learned.loc[i, 'lapse_high'] = fit_result.loc[0, 'lapsehigh']
    learned.loc[i, 'time_zone'] = time_zone_number

# Drop mice with faulty RT
learned = learned[learned['reaction_time'].notnull()]
learned['lab_number'] = learned.lab.map(institution_map()[0])
learned = learned.sort_values('lab_number')

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

# Prepare decoding data
decoding_result = pd.DataFrame(columns=['original', 'original_shuffled', 'confusion_matrix',
                                        'control', 'control_shuffled', 'control_cm'])
decod = learned.copy()
decoding_set = decod[METRICS].values
control_set = decod[METRICS_CONTROL].values

# Prepare lab labels for subselection of a fixed amount of mice per lab
labels = np.array(decod['lab_number'])
labels_nr = np.arange(labels.shape[0])
labels_decod = np.ravel([[lab] * N_MICE for i, lab in enumerate(np.unique(labels))])
labels_shuffle = np.ravel([[lab] * N_MICE for i, lab in enumerate(np.unique(labels))])

# Generate random states for each iteration with a fixed seed
np.random.seed(424242)

# Loop over iterations of random draws of mice
for i in range(ITERATIONS):
    if np.mod(i+1, 100) == 0:
        print('Iteration %d of %d' % (i+1, ITERATIONS))

    # Randomly select N mice from each lab to equalize classes
    use_index = np.empty(0, dtype=int)
    for j, lab in enumerate(np.unique(labels)):
        use_index = np.concatenate([use_index, np.random.choice(labels_nr[labels == lab],
                                                               N_MICE, replace=False)])

    # Original data
    decoding_result.loc[i, 'original'], conf_matrix = decoding(decoding_set[use_index],
                                                               labels_decod, clf)
    decoding_result.loc[i, 'confusion_matrix'] = (conf_matrix
                                                  / conf_matrix.sum(axis=1)[:, np.newaxis])

    # Shuffled data
    np.random.shuffle(labels_shuffle)
    decoding_result.loc[i, 'original_shuffled'], _ = decoding(decoding_set[use_index],
                                                              labels_shuffle, clf)

    # Positive control data
    decoding_result.loc[i, 'control'], conf_matrix = decoding(control_set[use_index],
                                                              labels_decod, clf)
    decoding_result.loc[i, 'control_cm'] = (conf_matrix
                                            / conf_matrix.sum(axis=1)[:, np.newaxis])

    # Shuffled positive control data
    np.random.shuffle(labels_shuffle)
    decoding_result.loc[i, 'control_shuffled'], _ = decoding(control_set[use_index],
                                                             labels_shuffle, clf)

# Save to csv
decoding_result.to_pickle(join(datapath(), 'classification_results',
                               'classification_results_basic_%s.pkl' % DECODER))
