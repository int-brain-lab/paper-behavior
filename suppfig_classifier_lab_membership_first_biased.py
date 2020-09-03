#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Decode in which lab a mouse was trained based on its behavioral metrics during the first three
biased sessions, regardless of when the mice reached proficiency.

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
from paper_behavior_functions import institution_map, QUERY
from dj_tools import fit_psychfunc, dj2pandas
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import f1_score, confusion_matrix

# Settings
DECODER = 'bayes'           # forest, bayes or regression
N_MICE = 5                  # how many mice per lab to sub-sample
ITERATIONS = 2000           # how often to decode
METRICS = ['perf_easy', 'threshold_l', 'threshold_r', 'threshold_n', 'bias_l', 'bias_r', 'bias_n']
METRICS_CONTROL = ['perf_easy', 'threshold_l', 'threshold_r', 'threshold_n',
                   'bias_l', 'bias_r', 'bias_n', 'time_zone']


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
    use_sessions, _ = query_sessions_around_criterion(criterion='biased',
                                                      days_from_criterion=[1, 3])
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
    neutral_fit = fit_psychfunc(behav[(behav['subject_nickname'] == nickname)
                                    & (behav['probabilityLeft'] == 50)])
    perf_easy = (behav.loc[behav['subject_nickname'] == nickname, 'correct_easy'].mean()) * 100

    fits = pd.DataFrame(data={'perf_easy': perf_easy,
                              'threshold_l': left_fit['threshold'],
                              'threshold_r': right_fit['threshold'],
                              'threshold_n': neutral_fit['threshold'],
                              'bias_l': left_fit['bias'],
                              'bias_r': right_fit['bias'],
                              'bias_n': neutral_fit['bias'],
                              'time_zone': time_zone_number,
                              'nickname': nickname, 'lab_number': lab})
    biased_fits = biased_fits.append(fits, sort=False)

    # Remove mice that did not have a 50:50 block
    biased_fits = biased_fits[biased_fits['threshold_n'].notnull()]

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
decoding_result.to_pickle(join('classification_results',
                               'classification_results_first_biased_%s.pkl' % DECODER))
