# -*- coding: utf-8 -*-
"""
Query the number of mice at different timepoints of the pipeline
@author 22 April 2020, Anne Urai
"""
from ibl_pipeline import subject, reference, behavior


dates = ['2019-01-01', '2019-03-01', '2020-01-01', '2020-04-01']
dates = ['2019-01-01', '2019-05-01', '2019-11-01', '2020-04-01']

print('All mice in database:')
for d in dates:

    # which mice were in the database by then?
    subj_query = subject.Subject * subject.SubjectLab * reference.Lab * \
        behavior.TrialSet & 'session_start_time < "%s"'%d
    subj_df = subj_query.fetch(format='frame').reset_index()

    print('%s, %d mice, %d labs, %d choices'%(d, subj_df.subject_uuid.nunique(),
                                              subj_df.lab_name.nunique(),
                                              subj_df.n_trials.sum()))

print('Brainwide map project mice:')
for d in dates:

    # which mice were in the database by then?
    subj_query = subject.Subject * subject.SubjectLab * reference.Lab \
                 * (subject.SubjectProject & 'subject_project = "ibl_neuropixel_brainwide_01"') \
                 * (behavior.TrialSet & 'session_start_time < "%s"'%d)
    subj_df = subj_query.fetch(format='frame').reset_index()

    print('%s, %d mice, %d labs, %d choices'%(d, subj_df.subject_uuid.nunique(),
                                              subj_df.lab_name.nunique(),
                                              subj_df.n_trials.sum()))
