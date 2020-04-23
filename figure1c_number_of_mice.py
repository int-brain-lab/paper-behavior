# -*- coding: utf-8 -*-
"""
Query the number of mice at different timepoints of the pipeline

@author: Anne Urai & Guido Meijer, 16 Jan 2020
Updated 22 April 2020, Anne Urai
"""

from paper_behavior_functions import query_subjects
from ibl_pipeline import subject, acquisition, reference, behavior
from ibl_pipeline.analyses import behavior as behavior_analysis
import pandas as pd

# =========================
# 1. Query all mice on brainwide map project
# =========================

all_mice = subject.Subject * subject.SubjectLab * reference.Lab * \
            subject.SubjectProject() & 'subject_project = "ibl_neuropixel_brainwide_01"'
print('1. Total # of mice in brainwide project: %d' % len(all_mice))

# ==================================================
# Exclude mice that are still in training at the date of cutt-off
# ==================================================

still_training = all_mice * subject.Subject.aggr(behavior_analysis.SessionTrainingStatus,
                                        session_start_time='max(session_start_time)')\
                   * behavior_analysis.SessionTrainingStatus - subject.Death \
                    & 'training_status = "in_training"' & 'session_start_time > "2020-03-01"'

# ==================================================
# Get mice that started training
# ==================================================

mice_started_training = (all_mice & (acquisition.Session() & 'task_protocol LIKE "%training%"'))
print('2. Number of mice that went into training: %d'% len(mice_started_training))
print('3. Number of mice that are still in training (exclude from 1 and 2): %d' % len(still_training))

# ==================================================
# Mice that reached trained
# ==================================================

print('4. Number of mice that reached trained: %d' % len(query_subjects()))
print('5. Number of mice that reached ready4ephysrig: %d' % len(query_subjects(criterion='ready4ephysrig')))

print(pd.DataFrame(still_training))
