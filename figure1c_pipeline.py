#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 14 18:26 2019

Measure the number of animals at each stage of the pipeline.

@author: Gaelle Chapuis
edited by Anne Urai
"""

from ibl_pipeline import subject, reference, acquisition, behavior  # noqa
from ibl_pipeline.analyses import behavior as behavior_analysis
from paper_behavior_functions import query_subjects
import datetime


def get_mouse_n(subj_query):
    """
    Get number of mice from subject query
    Ensure date is set appropriately.
    """
    n = len(subj_query.fetch(format='frame').reset_index()['subject_nickname'].unique())
    return n

# ============================================= #
# 1. all mice that had a headimplant
# - listed as project neuropixels_brainwide
# - dead or alive
# - born before
# ============================================= #

# Query all subjects with project ibl_neuropixel_brainwide_01
query = (subject.Subject * subject.SubjectLab * reference.Lab * subject.SubjectProject
              & 'subject_project = "ibl_neuropixel_brainwide_01"')

# remove those that were born less than 3 months ago and that are still alive
today = datetime.datetime.today().strftime('%Y-%m-%dT%H:%M')
recovery_time = datetime.datetime.today() - datetime.timedelta(days=7*13)
young_mice = (query & 'subject_birth_date > "%s"'%recovery_time) - subject.Death()

implanted_subjects = get_mouse_n(query) - get_mouse_n(young_mice)
print('Implanted: %d'%implanted_subjects)

# ============================================= #
# 2. all mice that started habituationChoiceWorld
# ============================================= #

# # Query all subjects with project ibl_neuropixel_brainwide_01
# query2 = query * acquisition.Session & 'task_protocol LIKE "%habituation%"'
#
# # # figure out which subjects may have died in surgery
# # query1_fetch = query.fetch(format='frame').reset_index()
# # query2_fetch = query2.fetch(format='frame').reset_index()
# # dead_sjs = list(set(query1_fetch.subject_nickname.unique()) - set(query2_fetch.subject_nickname.unique()))
# # dead_query = query1_fetch[query1_fetch['subject_nickname'].isin(dead_sjs)]
#
# started_subjects = get_mouse_n(query2)
# print('HabituationChoiceWorld: %d'%started_subjects)

# Query all subjects with project ibl_neuropixel_brainwide_01
query3 = query * acquisition.Session & 'task_protocol LIKE "%training%"'
started_subjects = get_mouse_n(query3)
print('TrainingChoiceWorld: %d'%started_subjects)

# ============================================= #
# 3. all mice that reached trained
# ============================================= #

trained_subjects = get_mouse_n(query_subjects())
print('Trained: %d'%trained_subjects)

# ============================================= #
# 3. all mice that reached trained
# ============================================= #

trained_subjects = get_mouse_n(query_subjects())
print('Trained: %d'%trained_subjects)


#
#
# #  TIP - use these commands to get all status types:
# #   import datajoint as dj
# #   dj.U('training_status') & behavior_analysis.SessionTrainingStatus
#
# #  -- End of pipeline (trained mice)
# subj_trained_query = query_subjects(as_dataframe=False)
# print(get_n_mouse(subj_query=subj_trained_query))
# # n_live_trained = (subj_trained_query - subject.Death).fetch(format='frame').shape[0]
#
# '''
# #  -- Begining of pipeline (all mice in project)
# subj_total_query = subject.Subject * subject.SubjectProject & \
#     'subject_project = "ibl_neuropixel_brainwide_01"'
# #  I have not found how to restrict for mice that were entered in the database prior to date
# '''
#
# #  -- Middle of pipeline (mice that entered training)
#
# subj_intraining_query = (subject.Subject * subject.SubjectProject &
#                          'subject_project = "ibl_neuropixel_brainwide_01"').aggr(
#                              (acquisition.Session * behavior_analysis.SessionTrainingStatus()) &
#                              'training_status="in_training"', 'subject_nickname',
#                              'training_status', session_start_time='max(session_start_time)')
# print(get_n_mouse(subj_query=subj_intraining_query))
#
#
# subjects = subject.Subject * subject.SubjectProject & 'subject_project = "ibl_neuropixel_brainwide_01"'
# last_sessions = subjects.aggr(acquisition.Session & 'task_protocol!="NULL"', 'subject_nickname', 'task_protocol', session_start_time='max(session_start_time)')
# last_sessions * behavior_analysis.SessionTrainingStatus
#
#
# subjects = subject.Subject * subject.SubjectProject & 'subject_project = "ibl_neuropixel_brainwide_01"'
# last_sessions = subjects.aggr(acquisition.Session & 'task_protocol is not NULL', 'subject_nickname', session_start_time='max(session_start_time)')
# last_sessions * acquisition.Session.proj('task_protocol') * behavior_analysis.SessionTrainingStatus & 'subject_nickname="SWC_002"'
#
# # -- FROM SHAN
# # 2
# sessions = acquisition.Session & 'session_start_time < "2019-09-30"'
# all_animals = (subject.Subject * subject.SubjectProject & \
#     'subject_project="ibl_neuropixel_brainwide_01"' & 'subject_ts<"2019-09-30"')
# all_animals_with_last_sessions = all_animals.aggr(sessions, session_start_time = 'max(session_start_time)')
# all_in_training = all_animals_with_last_sessions * behavior_analysis.SessionTrainingStatus & \
#     'training_status="in_training"'
# alive_in_training = all_in_training - subject.Death
# dead_in_training = all_in_training & subject.Death
#
# # 1
# all_animals = (subject.Subject * subject.SubjectProject & \
#     'subject_project="ibl_neuropixel_brainwide_01"' & 'subject_ts<"2019-10-01"') - acquisition.Session
#
# alive_animals = all_animals - subject.Death
