#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 14 18:26 2019

Measure the number of animals at each stage of the pipeline.

@author: Gaelle Chapuis
"""

from ibl_pipeline import subject, reference, acquisition  # noqa
from ibl_pipeline.analyses import behavior as behavior_analysis
from paper_behavior_functions import query_subjects


def get_n_mouse(subj_query, date_criterion='date_trained < "2019-09-30"'):
    """
    Get number of mice from subject query
    Ensure date is set appropriately.
    """
    number_mice = (subj_query & date_criterion).fetch(format='frame').shape[0]

    return number_mice


#  TIP - use these commands to get all status types:
#   import datajoint as dj
#   dj.U('training_status') & behavior_analysis.SessionTrainingStatus

#  -- End of pipeline (trained mice)
subj_trained_query = query_subjects(as_dataframe=False)
print(get_n_mouse(subj_query=subj_trained_query))
# n_live_trained = (subj_trained_query - subject.Death).fetch(format='frame').shape[0]

'''
#  -- Begining of pipeline (all mice in project)
subj_total_query = subject.Subject * subject.SubjectProject & \
    'subject_project = "ibl_neuropixel_brainwide_01"'
#  I have not found how to restrict for mice that were entered in the database prior to date
'''

#  -- Middle of pipeline (mice that entered training)

subj_intraining_query = (subject.Subject * subject.SubjectProject &
                         'subject_project = "ibl_neuropixel_brainwide_01"').aggr(
                             (acquisition.Session * behavior_analysis.SessionTrainingStatus()) &
                             'training_status="in_training"', 'subject_nickname',
                             'training_status', session_start_time='max(session_start_time)')
print(get_n_mouse(subj_query=subj_intraining_query))


subjects = subject.Subject * subject.SubjectProject & 'subject_project = "ibl_neuropixel_brainwide_01"'
last_sessions = subjects.aggr(acquisition.Session & 'task_protocol!="NULL"', 'subject_nickname', 'task_protocol', session_start_time='max(session_start_time)')
last_sessions * behavior_analysis.SessionTrainingStatus


subjects = subject.Subject * subject.SubjectProject & 'subject_project = "ibl_neuropixel_brainwide_01"'
last_sessions = subjects.aggr(acquisition.Session & 'task_protocol is not NULL', 'subject_nickname', session_start_time='max(session_start_time)')
last_sessions * acquisition.Session.proj('task_protocol') * behavior_analysis.SessionTrainingStatus & 'subject_nickname="SWC_002"'

# -- FROM SHAN
# 2
sessions = acquisition.Session & 'session_start_time < "2019-09-30"'
all_animals = (subject.Subject * subject.SubjectProject & \
    'subject_project="ibl_neuropixel_brainwide_01"' & 'subject_ts<"2019-09-30"')
all_animals_with_last_sessions = all_animals.aggr(sessions, session_start_time = 'max(session_start_time)')
all_in_training = all_animals_with_last_sessions * behavior_analysis.SessionTrainingStatus & \
    'training_status="in_training"'
alive_in_training = all_in_training - subject.Death
dead_in_training = all_in_training & subject.Death

# 1 
all_animals = (subject.Subject * subject.SubjectProject & \
    'subject_project="ibl_neuropixel_brainwide_01"' & 'subject_ts<"2019-10-01"') - acquisition.Session

alive_animals = all_animals - subject.Death
dead_animals = all_animals & subject.Death)