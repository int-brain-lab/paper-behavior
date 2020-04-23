# -*- coding: utf-8 -*-
"""
Query the number of mice at different timepoints of the pipeline

@author: Anne Urai & Guido Meijer, 16 Jan 2020
Updated 22 April 2020, Anne Urai
"""

from paper_behavior_functions import query_subjects
from ibl_pipeline import subject, acquisition, reference, behavior
from ibl_pipeline.analyses import behavior as behavior_analysis

# =========================
# 1. Query all mice on brainwide map project
# =========================

all_mice = subject.Subject * subject.SubjectLab * reference.Lab * \
            subject.SubjectProject() & 'subject_project = "ibl_neuropixel_brainwide_01"'
print('Total # of mice in brainwide project: %d' % len(all_mice))

# ==================================================
# Exclude mice that are still in training at the date of cutt-off
# ==================================================

all_mice_df = all_mice.fetch(format='frame').reset_index()

still_training = []
# for each mouse, get its latest training status
for m in all_mice_df.subject_uuid:
  subj_query = (subject.Subject & dict(subject_uuid=m))
  last_session = subj_query.aggr(
      behavior.TrialSet, session_start_time='max(session_start_time)')

# what was the last session of this mouse?
  if not len(last_session):
    training_status = 'no_data'
    last_session_date = None
  else:
     training_status = behavior_analysis.SessionTrainingStatus & last_session
     if len(training_status):
        training_status = (training_status).fetch1('training_status')
        last_session_date = last_session.fetch1('session_start_time')

        # do we exclude this animal, as it's still in training?
        # still count it if it has died
        if training_status in 'in_training' \
          and not len(subj_query & subject.Death):
            nickname = subj_query.fetch1('subject_nickname')
            still_training.append({'subject':nickname, 'last_training':last_session_date})

print('Number of mice that are still in training: %d' % len(still_training))
print(still_training)

# ==================================================
# Get mice that started training
# ==================================================

mice_started_training = (all_mice & (acquisition.Session() & 'task_protocol LIKE "%training%"'))
print('Number of mice that went into training: %d' \
      % len(mice_started_training) - len(still_training))

# ==================================================
# Mice that reached trained
# ==================================================

print('Number of mice that reached trained: %d' % len(query_subjects()))
print('Number of mice that reached ready4ephysrig: %d' % len(query_subjects(criterion='ready4ephysrig')))
