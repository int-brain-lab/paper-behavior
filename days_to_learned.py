# -*- coding: utf-8 -*-
"""
Created on Thu Oct 10 2019

Calculate average number of days to reach criterion 1a

@author: miles
"""

import numpy as np
from ibl_pipeline import acquisition, behavior, action
from ibl_pipeline.analyses import behavior as behavior_analysis
from paper_behavior_functions import query_subjects

subj = query_subjects()  # get subjects
subj_crit = subj.aggr(
                     (acquisition.Session * behavior_analysis.SessionTrainingStatus)
                     & 'training_status="trained_1a"',
                    'subject_nickname', date_criterion='min(date(session_start_time))')
to_trained = subj_crit * acquisition.Session & 'session_start_time <= date_criterion'

# Calculate the number of sessions to 1a
sess_to_trained = subj.aggr(to_trained.proj('session_uuid'), n='COUNT(*)').fetch('n')

# Calculate the number of days (including weekends) to trained
days_to_trained = subj.aggr(to_trained.proj('date_criterion', 'session_start_time'),
                            n='DATEDIFF(date_criterion, MIN(DATE(session_start_time)))').fetch('n')

# Calculate days from first water restriction
restriction = to_trained.proj('date_criterion', 'session_start_time') * \
              action.WaterRestriction & 'restriction_start_time < MIN(session_start_time)'
days_from_water_restriction = subj.aggr(restriction,
                                        n='DATEDIFF(date_criterion, '
                                          'MIN(DATE(restriction_start_time)))').fetch('n')

print('sessions to learned: mean = {}; std = {}; n = {}'.format(
    np.round(np.mean(sess_to_trained)),
    np.round(np.std(sess_to_trained)),
    sess_to_trained.__len__()))

print('days to learned (incl. weekends): mean = {}; std = {}; n = {}'.format(
    np.round(np.mean(days_to_trained)),
    np.round(np.std(days_to_trained)),
    days_to_trained.__len__()))

# Some info about trial number
last = acquisition.Session() * subj_crit & 'date(session_start_time) between date_criterion - 2 and date_criterion'
count = last.aggr(behavior.TrialSet.Trial, n='COUNT(*)')
trials = subj.proj('subject_uuid', 'subject_nickname').aggr(count, avg_n='avg(n)').fetch('avg_n')
trials = np.array([float(f) for f in trials])

print('mean = {}; std = {}; n = {}'.format(
    np.round(np.mean(trials)),
    np.round(np.std(trials)),
    trials.__len__()))