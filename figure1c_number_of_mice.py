# -*- coding: utf-8 -*-
"""
Query the number of mice at different timepoints of the pipeline

@author: Anne Urai & Guido Meijer
16 Jan 2020
"""

from paper_behavior_functions import query_subjects
from ibl_pipeline import subject, acquisition, reference
from ibl_pipeline.analyses import behavior as behavior_analysis

# Query all mice on brainwide map project
all_mice = subject.SubjectProject() & 'subject_project = "ibl_neuropixel_brainwide_01"'

# Exclude mice that were still in training at the date of cutt-off
subj_query = (subject.Subject * subject.SubjectLab * reference.Lab * subject.SubjectProject
              & 'subject_project = "ibl_neuropixel_brainwide_01"').aggr(
    (acquisition.Session * behavior_analysis.SessionTrainingStatus())
    & 'training_status="in_training"',
    'subject_nickname', 'sex', 'subject_birth_date', 'institution_short',
    last_training_session='max(date(session_start_time))')
mice_in_training = (subj_query & 'last_training_session > "2019-11-30"')
mice_at_start = len(all_mice)-len(mice_in_training)
print('Number of mice at start: %d' % mice_at_start)

# Get mice that when into training
mice_training = (subj_query & 'last_training_session < "2019-11-30"')

# Get dropout during habituation
training_query = (subject.Subject * subject.SubjectLab * reference.Lab * subject.SubjectProject
                  & 'subject_project = "ibl_neuropixel_brainwide_01"').aggr(
    acquisition.Session() & 'task_protocol LIKE "%training%"',
    num_sessions='count(session_start_time)').fetch(format='frame')
print('Number of mice that went into habituation: %d' % (len(mice_training)
                                                         + sum(
                                                             training_query['num_sessions'] == 0)))
print('Number of mice that went into training: %d' % len(mice_training))

# Get number of mice that reached trained
print('Number of mice that reached trained: %d' % len(query_subjects()))

print('Number of mice that are still in training: %d' % len(mice_in_training))

# Get number of mice ready for ephys
subj_query = (query_subjects().proj('subject_uuid') * subject.Subject * subject.SubjectProject
              & 'subject_project = "ibl_neuropixel_brainwide_01"').aggr(
    (acquisition.Session * behavior_analysis.SessionTrainingStatus())
    & 'training_status="ready4ephysrig"',
    'subject_nickname', 'sex', 'subject_birth_date',
    date_trained='min(date(session_start_time))')
print('Number of mice that reached ready for ephys: %d' % len(subj_query))
