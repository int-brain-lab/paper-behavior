#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Decode in which lab a mouse was trained based on its behavioral metrics during the three sessions
of the full task variant in which the mouse was determined to be ready for ephys.

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

import numpy as np
from os.path import join
from paper_behavior_functions import \
    institution_map, QUERY, fit_psychfunc, dj2pandas, load_csv, datapath
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import f1_score, confusion_matrix

# Settings
DECODER = 'bayes'           # bayes, forest or regression
N_MICE = 8                  # how many mice per lab to randomply sub-sample
ITERATIONS = 2000           # how often to decode with random sub-samples
METRICS = ['threshold_l', 'threshold_r', 'bias_l', 'bias_r', 'lapselow_l', 'lapselow_r',
           'lapsehigh_l', 'lapsehigh_r']
METRICS_CONTROL = ['threshold_l', 'threshold_r', 'bias_l', 'bias_r', 'lapselow_l', 'lapselow_r',
                   'lapsehigh_l', 'lapsehigh_r', 'time_zone']


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
    behav = load_csv('Fig4.csv')

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
                              'nickname': nickname, 'lab_number': lab,
                              'time_zone': time_zone_number})
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

# Prepare decoding data
decoding_result = pd.DataFrame(columns=['original', 'original_shuffled', 'confusion_matrix',
                                        'control', 'control_shuffled', 'control_cm'])
decod = biased_fits.copy()
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
decoding_result.to_pickle(join(datapath(), 'classification_results_full_%s.pkl' % DECODER))
