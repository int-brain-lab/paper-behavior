#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Create list of UUIDs used in the paper

@author: Alex Pan
"""

import numpy as np
from ibl_pipeline import reference, subject, acquisition, behavior
from ibl_pipeline.analyses import behavior as behavior_analysis
from paper_behavior_functions import query_subjects, query_sessions_around_criterion

# All subjects
subjects = query_subjects(as_dataframe=True)

# Fig2b_time_to_trained
# Query sessions
subjects = query_subjects(as_dataframe=True)
# Create dataframe with behavioral metrics of all mice
ses_2b = []
for i, nickname in enumerate(subjects['subject_nickname']):
    if np.mod(i+1, 10) == 0:
        print('Loading data of subject %d of %d' % (i+1, len(subjects['subject_nickname'])))

    # Get sessions and trials which are flagged as in_training
    ses_start = (acquisition.Session * subject.Subject * behavior_analysis.SessionTrainingStatus
                 & ('subject_nickname = "%s"' % nickname)
                 & 'training_status = "in_training" OR training_status = "untrainable"').proj(
                         'session_uuid').fetch(format='frame')
    sess = ses_start['session_uuid'].unique()
    if i == 0:
        ses_2b = sess
    else:
        ses_2b = np.append(ses_2b, sess)

np.save('fig2b_uuids.npy', ses_2b)

# Figure4b
use_sessions, use_days = query_sessions_around_criterion(criterion='ephys',
                                                         days_from_criterion=[2, 0],
                                                         as_dataframe=False)
# restrict by list of dicts with uuids for these sessions
b = (use_sessions & 'task_protocol LIKE "%biased%"') \
    * subject.Subject * subject.SubjectLab * reference.Lab * \
    behavior.TrialSet
# reduce the size of the fetch
b2 = b.proj('session_uuid')
bdat = b2.fetch(format='frame').reset_index()
fig4b = bdat['session_uuid']
np.save('fig4b_uuids.npy', fig4b)
