# -*- coding: utf-8 -*-
"""
Query the number of mice at different timepoints of the pipeline

@author: Anne Urai, Guido Meijer, Miles Wells, 16 Jan 2020
Updated 22 April 2020, Anne Urai
"""

from paper_behavior_functions import query_subjects, CUTOFF_DATE
from ibl_pipeline import subject, acquisition, reference
from ibl_pipeline.analyses import behavior as behavior_analysis


# =========================
# 1. Query all mice on brainwide map project which began training before the paper's cutoff date
# =========================

all_mice = (subject.Subject * subject.SubjectLab * reference.Lab * subject.SubjectProject()
            & 'subject_project = "ibl_neuropixel_brainwide_01"').aggr(
    acquisition.Session, first_session='min(date(session_start_time))')

# Filter mice that started training after the paper's cutoff date
all_mice = all_mice.aggr(acquisition.Session, first_session='min(date(session_start_time))')
all_mice = (all_mice & 'first_session < "%s"' % CUTOFF_DATE)

print('1. Total # of mice in brainwide project: %d' % len(all_mice))

# ==================================================
# Exclude mice that are still in training at the date of cut-off, meaning they have not yet
# reached any learned criteria
# ==================================================

still_training = all_mice * subject.Subject.aggr(behavior_analysis.SessionTrainingStatus,
                                                 session_start_time='max(session_start_time)') \
                   * behavior_analysis.SessionTrainingStatus - subject.Death \
                   & 'training_status = "in_training"' & 'session_start_time > "%s"' % CUTOFF_DATE
# print(pd.DataFrame(still_training))

# ==================================================
# Get mice that started training
# ==================================================

mice_started_training = (all_mice & (acquisition.Session() & 'task_protocol LIKE "%training%"'))
print('2. Number of mice that went into training: %d' % len(mice_started_training))
print('3. Number of mice that are still in training (exclude from 1 and 2): %d' % len(still_training))

# ==================================================
# Mice that reached trained
# ==================================================

trained = query_subjects(criterion='trained')
print('4. Number of mice that reached trained: %d' % len(trained))
print('5. Number of mice that reached ready4ephys: %d' % len(query_subjects(criterion='ephys')))

# ==================================================
# Trained mice yet to meet final criterion at the cut off date.
# These mice did not quite reach ready4ephysrig by the cut-off date, but were likely to
# ==================================================

# mice that reached trained but not ready4ephys, didn't die before the cut-off, and had fewer
# than 40 sessions (no session marked as 'untrainable')
session_training_status = acquisition.Session * behavior_analysis.SessionTrainingStatus()
trained_not_ready = (trained.aggr(session_training_status,
                                  unfinished='SUM(training_status="ready4ephys" OR '
                                             'training_status="untrainable" OR '
                                             'training_status="unbiasable") = 0')
                            .aggr(subject.Death, 'unfinished',
                                  alive='death_date IS NULL OR death_date > "%s"' % CUTOFF_DATE,
                                  keep_all_rows=True))

print('6. Number of mice that remain in training at the time of writing: %d' %
      len(trained_not_ready & 'alive = True AND unfinished = True'))

