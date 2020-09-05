#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Queries to save static csv files with data for each figure

Guido Meijer
Jul 1, 2020
"""

from os import mkdir
from os.path import join, isdir
import pandas as pd
from paper_behavior_functions import (query_subjects, query_sessions_around_criterion,
                                      institution_map, CUTOFF_DATE, dj2pandas)
from ibl_pipeline.analyses import behavior as behavioral_analyses
from ibl_pipeline import reference, subject, behavior, acquisition

# Get map of lab number to institute
institution_map, _ = institution_map()

# create data directory if it doesn't exist yet
if not isdir('data'):
    mkdir('data')

# Create list of subjects used
subjects = query_subjects(as_dataframe=True)
subjects.to_csv(join('data', 'subjects.csv'))


# %%=============================== #
# FIGURE 2
# ================================= #
print('Starting figure 2..')

# Query list of subjects to use
use_subjects = query_subjects()

# Figure 2ab
b = (behavioral_analyses.BehavioralSummaryByDate * use_subjects)
behav = b.fetch(order_by='institution_short, subject_nickname, training_day',
                format='frame').reset_index()
behav['institution_code'] = behav.institution_short.map(institution_map)

# Save to csv
behav.to_csv(join('data', 'Fig2ab.csv'))

# Figure 2c
all_mice = (subject.Subject * subject.SubjectLab * reference.Lab
            * subject.SubjectProject() & 'subject_project = "ibl_neuropixel_brainwide_01"')
mice_started_training = (all_mice & (acquisition.Session() & 'task_protocol LIKE "%training%"'))
still_training = all_mice.aggr(behavioral_analyses.SessionTrainingStatus,
                               session_start_time='max(session_start_time)') \
                                    * behavioral_analyses.SessionTrainingStatus - subject.Death \
                                    & 'training_status = "in_training"' \
                                    & 'session_start_time > "%s"' % CUTOFF_DATE
use_subjects = mice_started_training - still_training

# Get training status and training time in number of sessions and trials
ses = (use_subjects
       * behavioral_analyses.SessionTrainingStatus
       * behavioral_analyses.PsychResults).proj(
               'subject_nickname', 'training_status', 'n_trials_stim', 'institution_short').fetch(
                                                                   format='frame').reset_index()
ses['n_trials'] = [sum(i) for i in ses['n_trials_stim']]
ses = ses.drop('n_trials_stim', axis=1)

# Save to csv
ses.to_csv(join('data', 'Fig2c.csv'))

# Figure 2d
ses = (use_subjects * behavioral_analyses.SessionTrainingStatus * behavioral_analyses.PsychResults
       & 'training_status = "in_training" OR training_status = "untrainable"').proj(
               'subject_nickname', 'n_trials_stim', 'institution_short').fetch(format='frame')
ses = ses.reset_index()
ses['n_trials'] = [sum(i) for i in ses['n_trials_stim']]

# Construct dataframe
training_time = pd.DataFrame(columns=['sessions'], data=ses.groupby('subject_nickname').size())
training_time['trials'] = ses.groupby('subject_nickname').sum()
training_time['lab'] = ses.groupby('subject_nickname')['institution_short'].apply(list).str[0]

# Change lab name into lab number
training_time['lab_number'] = training_time.lab.map(institution_map)
training_time = training_time.sort_values('lab_number')

# Save to csv
training_time.to_csv(join('data', 'Fig2d.csv'))

# %%=============================== #
# FIGURE 3
# ================================= #
print('Starting figure 3..')

# query sessions
use_sessions, _ = query_sessions_around_criterion(criterion='trained',
                                                  days_from_criterion=[2, 0],
                                                  as_dataframe=False,
                                                  force_cutoff=True)
use_sessions = use_sessions & 'task_protocol LIKE "%training%"'  # only get training sessions

# query all trials for these sessions, it's split in two because otherwise the query would become
# too big to handle in one go
b = (use_sessions * subject.Subject * subject.SubjectLab * reference.Lab
     * behavior.TrialSet.Trial)

# reduce the size of the fetch
b2 = b.proj('institution_short', 'subject_nickname', 'task_protocol', 'session_uuid',
            'trial_stim_contrast_left', 'trial_stim_contrast_right', 'trial_response_choice',
            'task_protocol', 'trial_stim_prob_left', 'trial_feedback_type',
            'trial_response_time', 'trial_stim_on_time')

# construct pandas dataframe
bdat = b2.fetch(order_by='institution_short, subject_nickname, session_start_time, trial_id',
                format='frame').reset_index()
behav = dj2pandas(bdat)
behav['institution_code'] = behav.institution_short.map(institution_map)

# save to disk
behav.to_csv(join('data', 'Fig3.csv'))

# %%=============================== #
# FIGURE 4
# ================================= #
print('Starting figure 4..')

# query sessions
use_sessions, _ = query_sessions_around_criterion(criterion='ephys',
                                                  days_from_criterion=[2, 0],
                                                  force_cutoff=True)
use_sessions = use_sessions & 'task_protocol LIKE "%biased%"'  # only get biased sessions

# restrict by list of dicts with uuids for these sessions
b = (use_sessions * subject.Subject * subject.SubjectLab * reference.Lab
     * behavior.TrialSet.Trial)

# reduce the size of the fetch
b2 = b.proj('institution_short', 'subject_nickname', 'task_protocol', 'session_uuid',
            'trial_stim_contrast_left', 'trial_stim_contrast_right', 'trial_response_choice',
            'task_protocol', 'trial_stim_prob_left', 'trial_feedback_type',
            'trial_response_time', 'trial_stim_on_time')

# construct pandas dataframe
bdat = b2.fetch(order_by='institution_short, subject_nickname, session_start_time, trial_id',
                format='frame').reset_index()
behav = dj2pandas(bdat)
behav['institution_code'] = behav.institution_short.map(institution_map)

# save to disk
behav.to_csv(join('data', 'Fig4.csv'))

# %%=============================== #
# FIGURE 5
# ================================= #
print('Starting figure 5..')

# Query sessions biased data
use_sessions, _ = query_sessions_around_criterion(criterion='biased',
                                                  days_from_criterion=[2, 3],
                                                  as_dataframe=False,
                                                  force_cutoff=True)


# restrict by list of dicts with uuids for these sessions
b = (use_sessions * subject.Subject * subject.SubjectLab * reference.Lab
     * behavior.TrialSet.Trial)

# reduce the size of the fetch
b2 = b.proj('institution_short', 'subject_nickname', 'task_protocol',
            'trial_stim_contrast_left', 'trial_stim_contrast_right',
            'trial_response_choice', 'task_protocol', 'trial_stim_prob_left',
            'trial_feedback_type')
bdat = b2.fetch(order_by='institution_short, subject_nickname, session_start_time, trial_id',
                format='frame').reset_index()
behav = dj2pandas(bdat)
behav['institution_code'] = behav.institution_short.map(institution_map)

# save to disk
behav.to_csv(join('data', 'Fig5.csv'))

# %%=============================== #
# FIGURE 3 - SUPPLEMENT 2
# ================================= #
print('Starting figure 3 - supplement 2..')

# Query sessions biased data
use_sessions, _ = query_sessions_around_criterion(
    criterion='biased',
    days_from_criterion=[-1, 3],
    force_cutoff=True)
use_sessions = use_sessions & 'task_protocol LIKE "%biased%"'  # only get biased sessions

# restrict by list of dicts with uuids for these sessions
b = (use_sessions * subject.Subject * subject.SubjectLab * reference.Lab
     * behavior.TrialSet.Trial)

# reduce the size of the fetch
b2 = b.proj('institution_short', 'subject_nickname', 'task_protocol',
            'trial_stim_contrast_left', 'trial_stim_contrast_right',
            'trial_response_choice', 'task_protocol', 'trial_stim_prob_left',
            'trial_feedback_type')
bdat = b2.fetch(order_by='institution_short, subject_nickname, session_start_time, trial_id',
                format='frame').reset_index()
behav = dj2pandas(bdat)
behav['institution_code'] = behav.institution_short.map(institution_map)

# save to disk
behav.to_csv(join('data', 'Fig3-supp2.csv'))
