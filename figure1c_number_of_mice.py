# -*- coding: utf-8 -*-
"""
Created on Thu Dec 19 16:15:10 2019

@author: guido
"""

import datetime
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
print('Number of mice still in training: %d' % len(mice_in_training))

# Get dropout after implantation
mice_trained = (subj_query & 'last_training_session < "2019-11-30"')
print('Number of mice that went into training: %d' % len(mice_trained))

# Get number of mice that reached trained
print('Number of mice that reached trained: %d' % len(query_subjects()))

# Get number of mice ready for ephys
subj_query = (query_subjects().proj('subject_uuid') * subject.Subject * subject.SubjectProject
              & 'subject_project = "ibl_neuropixel_brainwide_01"').aggr(
        (acquisition.Session * behavior_analysis.SessionTrainingStatus())
        & 'training_status="ready4ephysrig"',
        'subject_nickname', 'sex', 'subject_birth_date',
        date_trained='min(date(session_start_time))')
print('Number of mice that reached ready for ephys: %d' % len(subj_query))
