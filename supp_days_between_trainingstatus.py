"""
The International Brain Laboratory
Anne Urai, CSHL, 2020-09-07

Starting from reaching 1a/1b, show distributions of days to next training stages

"""
from ibl_pipeline import subject, reference
from ibl_pipeline.analyses import behavior as behavior_analysis

from paper_behavior_functions import QUERY

assert QUERY, 'This script requires a DataJoint instance, which was removed in Dec 2023.'

# Query all subjects with project ibl_neuropixel_brainwide_01 and get the date at which
# they reached a given training status
all_subjects = (subject.Subject * subject.SubjectLab * reference.Lab * subject.SubjectProject
                & 'subject_project = "ibl_neuropixel_brainwide_01"')
summ_by_date = all_subjects * behavior_analysis.BehavioralSummaryByDate
training_status_by_day = summ_by_date.aggr(behavior_analysis.SessionTrainingStatus(),
                                           daily_status='(training_status)')
df = (training_status_by_day
      .fetch(format='frame')
      .reset_index()
      .sort_values(by=['lab_name', 'session_date']))
print(df.daily_status.unique())
