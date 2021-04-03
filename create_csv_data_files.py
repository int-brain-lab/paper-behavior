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
                                      institution_map, CUTOFF_DATE, dj2pandas, datapath,
                                      query_session_around_performance)
from ibl_pipeline.analyses import behavior as behavioral_analyses
from ibl_pipeline import reference, subject, behavior, acquisition
import csv

# Get map of lab number to institute
institution_map, _ = institution_map()

# create data directory if it doesn't exist yet
root = datapath()
if not isdir(root):
    mkdir(root)

# Create list of subjects used
subjects = query_subjects(as_dataframe=True)
subjects.to_csv(join(root, 'subjects.csv'))


# %%=============================== #
# FIGURE 2
# ================================= #
print('Starting figure 2.')
# Figure 2af
use_subjects = query_subjects()
b = (behavioral_analyses.BehavioralSummaryByDate * use_subjects * behavioral_analyses.BehavioralSummaryByDate.PsychResults)
behav = b.fetch(order_by='institution_short, subject_nickname, training_day',
                format='frame').reset_index()
behav['institution_code'] = behav.institution_short.map(institution_map)
# Save to csv
behav.to_pickle(join(root, 'Fig2af.pkl'))

# Figure 2h
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
ses = (
    (use_subjects
     * behavioral_analyses.SessionTrainingStatus
     * behavioral_analyses.PsychResults)
    .proj('subject_nickname', 'training_status', 'n_trials_stim', 'institution_short')
    .fetch(format='frame')
    .reset_index()
)
ses['n_trials'] = [sum(i) for i in ses['n_trials_stim']]
ses = ses.drop('n_trials_stim', axis=1)

# Save to csv
ses.to_csv(join(root, 'Fig2d.csv'))

# Figure 2ab

# Query list of subjects to use
use_subjects = query_subjects()

b = (behavioral_analyses.BehavioralSummaryByDate * use_subjects)
behav = b.fetch(order_by='institution_short, subject_nickname, training_day',
                format='frame').reset_index()
behav['institution_code'] = behav.institution_short.map(institution_map)

# Save to csv
behav.to_csv(join(root, 'suppFig2_1.csv'))

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

# list of dicts - see https://int-brain-lab.slack.com/archives/CB13FQFK4/p1607369435116300 for explanation
sess = use_sessions.proj('task_protocol').fetch(format='frame').reset_index().to_dict('records')

# query all trials for these sessions, it's split in two because otherwise the query would become
# too big to handle in one go
b = (behavior.TrialSet.Trial & sess) \
    * subject.Subject * subject.SubjectLab * reference.Lab * acquisition.Session

# reduce the size of the fetch
b2 = b.proj('institution_short', 'subject_nickname',
            'trial_stim_contrast_left', 'trial_stim_contrast_right', 'trial_response_choice',
            'trial_stim_prob_left', 'trial_feedback_type', 'trial_response_time',
            'trial_stim_on_time', 'session_end_time', 'task_protocol', 'time_zone')

# construct pandas dataframe
bdat = b2.fetch(order_by='institution_short, subject_nickname, session_start_time, trial_id',
                format='frame').reset_index()
behav = dj2pandas(bdat)
behav['institution_code'] = behav.institution_short.map(institution_map)

# save to disk
behav.to_csv(join(root, 'Fig3.csv'))

# %%=============================== #
# FIGURE 4
# ================================= #
print('Starting figure 4..')

# query sessions
use_sessions, _ = query_sessions_around_criterion(criterion='ephys',
                                                  days_from_criterion=[2, 0],
                                                  force_cutoff=True)
use_sessions = use_sessions & 'task_protocol LIKE "%biased%"'  # only get biased sessions
# list of dicts - see https://int-brain-lab.slack.com/archives/CB13FQFK4/p1607369435116300 for explanation
sess = use_sessions.proj('task_protocol').fetch(format='frame').reset_index().to_dict('records')

# restrict by list of dicts with uuids for these sessions
b = (behavior.TrialSet.Trial & sess) \
    * subject.Subject * subject.SubjectLab * reference.Lab * acquisition.Session

# reduce the size of the fetch
b2 = b.proj('institution_short', 'subject_nickname', 'task_protocol', 'session_uuid',
            'trial_stim_contrast_left', 'trial_stim_contrast_right', 'trial_response_choice',
            'trial_stim_prob_left', 'trial_feedback_type',
            'trial_response_time', 'trial_stim_on_time', 'time_zone')

# construct pandas dataframe
bdat = b2.fetch(order_by='institution_short, subject_nickname, session_start_time, trial_id',
                format='frame').reset_index()
behav = dj2pandas(bdat)
behav['institution_code'] = behav.institution_short.map(institution_map)

# save to disk
behav.to_csv(join(root, 'Fig4.csv'))

# %%=============================== #
# FIGURE 5
# ================================= #
print('Starting figure 5..')

# Query sessions biased data
use_sessions, _ = query_sessions_around_criterion(criterion='biased',
                                                  days_from_criterion=[2, 3],
                                                  as_dataframe=False,
                                                  force_cutoff=True)
sess = use_sessions.proj('task_protocol').fetch(format='frame').reset_index().to_dict('records')

# restrict by list of dicts with uuids for these sessions
b = (behavior.TrialSet.Trial & use_sessions) \
    * acquisition.Session * subject.Subject * subject.SubjectLab * reference.Lab

# reduce the size of the fetch
b2 = b.proj('institution_short', 'subject_nickname', 'task_protocol',
            'trial_stim_contrast_left', 'trial_stim_contrast_right',
            'trial_response_choice', 'trial_stim_prob_left', 'trial_feedback_type')
bdat = b2.fetch(order_by='institution_short, subject_nickname, session_start_time, trial_id',
                format='frame').reset_index()
behav = dj2pandas(bdat)
behav['institution_code'] = behav.institution_short.map(institution_map)

# save to disk
behav.to_csv(join(root, 'Fig5.csv'))

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
sess = use_sessions.proj('task_protocol').fetch(format='frame').reset_index().to_dict('records')

# restrict by list of dicts with uuids for these sessions
b = (behavior.TrialSet.Trial & sess) \
    * acquisition.Session * subject.Subject * subject.SubjectLab * reference.Lab

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
behav.to_csv(join(root, 'Fig3-supp2.csv'))
behav = query_session_around_performance(perform_thres=0.8)
behav.to_pickle(join(root, 'suppfig_3-4af.pkl'))
