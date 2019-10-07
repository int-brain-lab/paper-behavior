# -*- coding: utf-8 -*-
"""
Created on Tue Sep  3 10:15:42 2019

Functions for behavioral paper analysis

@author: guido
"""

from ibl_pipeline import subject, acquisition, reference
import seaborn as sns
import os
import pandas as pd
from ibl_pipeline.analyses import behavior as behavior_analysis
# from IPython import embed as shell  # for debugging


def group_colors():
    return sns.color_palette("Set2", 7)


def institution_map():
    institution_map = {'Berkeley': 'Lab 1', 'CCU': 'Lab 2', 'CSHL': 'Lab 3', 'NYU': 'Lab 4',
            'Princeton': 'Lab 5', 'SWC': 'Lab 6', 'UCL': 'Lab 7'}
    col_names = ['AllLabs', 'Lab 1', 'Lab 2', 'Lab 3', 'Lab 4', 'Lab 5', 'Lab 6', 'Lab 7']

    return institution_map, col_names


def seaborn_style():
    """
    Set seaborn style for plotting figures
    """
    sns.set(style="ticks", context="paper", font_scale=1.2)
    sns.despine(trim=True)


def figpath():
    # Retrieve absolute path of paper-behavior dir
    repo_dir = os.path.dirname(os.path.realpath(__file__))
    # Make figure directory
    fig_dir = os.path.join(repo_dir, 'exported_figs')
    # Announce save location
    print('Figure save path: ' + fig_dir)
    # If doesn't already exist, create
    if not os.path.exists(fig_dir):
        os.mkdir(fig_dir)
    return fig_dir


def query_subjects(as_dataframe=False):
    """
    Query all mice for analysis of behavioral data

    Parameters
    ----------
    as_dataframe: boolean if true returns a pandas dataframe (default is False)
    """

    # Query all subjects with project ibl_neuropixel_brainwide_01 and get the date at which
    # they were flagged as trained_1a
    subj_query = (subject.Subject * subject.SubjectLab * reference.Lab * subject.SubjectProject
                  & ['subject_project = "ibl_neuropixel_brainwide_01"'
                     + ' OR subject_project = "ibl_retired"']).aggr(
        (acquisition.Session * behavior_analysis.SessionTrainingStatus())
        & 'training_status="trained_1a" OR training_status="trained_1b"',
        'subject_nickname', 'sex', 'subject_birth_date', 'institution_short',
        date_trained='min(date(session_start_time))')

    # Select subjects that reached trained_1a criterium before September 30th
    if as_dataframe is True:
        subjects = (
            subj_query & 'date_trained < "2019-09-30"').fetch(format='frame')
        subjects = subjects.sort_values(by=['lab_name']).reset_index()
    else:
        subjects = (subj_query & 'date_trained < "2019-09-30"')
    return subjects


def query_sessions(stable=False, as_dataframe=False):
    """
    Query all sessions for analysis of behavioral data

    Parameters
    ----------
    stable:          boolean if True only return sessions with stable hardware, which means
                     sessions after July 10, 2019 (default is False)
    as_dataframe:    boolean if True returns a pandas dataframe (default is False)
    """

    # Query sessions
    use_subjects = query_subjects().proj('subject_uuid')
    sessions = (acquisition.Session * subject.Subject * subject.SubjectLab * reference.Lab
                * use_subjects * behavior_analysis.SessionTrainingStatus
                & 'task_protocol LIKE "%training%" OR task_protocol LIKE "%biased%"').proj(
        'session_uuid', 'subject_uuid', 'subject_nickname', 'institution_short',
                        'task_protocol', 'training_status')

    # If required only output sessions with stable hardware
    if stable is True:
        sessions = sessions & 'date(session_start_time) > "2019-06-10"'

    # Transform into pandas Dataframe if requested
    if as_dataframe is True:
        sessions = sessions.fetch(
            order_by='institution_short, subject_nickname, session_start_time', format='frame')
        sessions = sessions.reset_index()

    return sessions


def query_sessions_around_criterion(criterion='trained', days_from_criterion=[3, 0],
                                    as_dataframe=False):
    """
    Query all sessions for analysis of behavioral data

    Parameters
    ----------
    criterion:              string indicating which criterion to use: trained, biased or ephys
    days_from_criterion:    two-element array which indicates which training days around the day
                            the mouse reached criterium to return, e.g. [3, 2] returns three days
                            before criterium reached up until 2 days after.
    as_dataframe:           return sessions as a pandas dataframe


    """

    # Query all included subjects
    use_subjects = query_subjects().proj('subject_uuid')

    # Get per subject the date at which the criterion is reached
    if criterion == 'trained':
        subj_crit = (subject.Subject * use_subjects).aggr(
                        (acquisition.Session * behavior_analysis.SessionTrainingStatus)
                        & 'training_status="trained_1a" OR training_status="trained_1b"',
                        'subject_nickname', date_criterion='min(date(session_start_time))').fetch(
                                format='frame')
    elif criterion == 'biased':
        subj_crit = (subject.Subject * use_subjects).aggr(
                (acquisition.Session * behavior_analysis.SessionTrainingStatus)
                & 'task_protocol LIKE "%biased%"',
                'subject_nickname', date_criterion='min(date(session_start_time))').fetch(
                        format='frame')
    elif criterion == 'ephys':
        subj_crit = (subject.Subject * use_subjects).aggr(
                        (acquisition.Session * behavior_analysis.SessionTrainingStatus)
                        & 'training_status="ready4ephysrig" OR training_status="ready4recording"',
                        'subject_nickname', date_criterion='min(date(session_start_time))').fetch(
                                format='frame')
    else:
        raise Exception('criterion must be trained, biased or ephys')
    subj_crit = subj_crit.reset_index()

    # Loop over subjects and get x days before and x days after reaching criterion
    # This part is currently very inefficient and should be improved!
    sessions = (subject.Subject * acquisition.Session
                & 'subject_nickname = "empty"').proj('subject_uuid')
    for i, nickname in enumerate(subj_crit['subject_nickname']):
        session_dates = (subject.Subject * behavior_analysis.BehavioralSummaryByDate
                         & 'subject_nickname = "%s"' % nickname).fetch(format='frame')
        session_dates = session_dates.reset_index()
        crit_ind = session_dates[
            session_dates['session_date'] == pd.Timestamp(
                    subj_crit.loc[subj_crit['subject_nickname'] == nickname,
                                  'date_criterion'].values[0])].index.values[0]
        from_date = session_dates.loc[crit_ind-(days_from_criterion[0]-1), 'session_date']
        to_date = session_dates.loc[crit_ind+(days_from_criterion[1]), 'session_date']
        crit_sessions = (subject.Subject * acquisition.Session
                         & 'subject_nickname = "%s"' % nickname
                         & 'date(session_start_time) >= "%s"' % from_date.date()
                         & 'date(session_start_time) <= "%s"' % to_date.date()).proj(
                                 'subject_uuid')
        sessions = sessions + crit_sessions

    if as_dataframe is True:
        sessions = sessions.fetch(format='frame')
        sessions = sessions.reset_index()

    return sessions
