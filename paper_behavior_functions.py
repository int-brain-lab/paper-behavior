# -*- coding: utf-8 -*-
"""
Created on Tue Sep  3 10:15:42 2019

Functions for behavioral paper analysis

@author: guido
"""

from ibl_pipeline import subject, acquisition, reference
import seaborn as sns
import os
import numpy as np
import pandas as pd
from ibl_pipeline.analyses import behavior as behavior_analysis


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
                  & 'subject_project = "ibl_neuropixel_brainwide_01"').aggr(
                          (acquisition.Session * behavior_analysis.SessionTrainingStatus())
                          & 'training_status="trained_1a" OR training_status="trained_1b"',
                          'subject_nickname', 'sex', 'subject_birth_date', 'institution',
                          date_trained='min(date(session_start_time))')

    # Select subjects that reached trained_1a criterium before September 30th
    if as_dataframe is True:
        subjects = (subj_query & 'date_trained < "2019-09-30"').fetch(format='frame')
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
                        'session_uuid', 'subject_uuid', 'subject_nickname', 'institution',
                        'task_protocol', 'training_status')

    # If required only output sessions with stable hardware
    if stable is True:
        sessions = sessions & 'date(session_start_time) > "2019-06-10"'

    # Transform into pandas Dataframe if requested
    if as_dataframe is True:
        sessions = sessions.fetch(format='frame')
        sessions = sessions.reset_index()

    return sessions


def query_sessions_around_criterium(days_from_trained=[3, 0]):
    """
    Query all sessions for analysis of behavioral data

    Parameters
    ----------
    days_from_trained: two-element array which indicates which training days around the day the
                       mouse reached criterium to return, e.g. [3, 2] returns three days before
                       criterium reached up untill 2 days after.
    """

    # Query all sessions including training status
    all_sessions = query_sessions(as_dataframe=True)

    # Loop through mice and find sessions around first trained session
    sessions = pd.DataFrame()
    for i, nickname in enumerate(np.unique(all_sessions['subject_nickname'])):

        # Get the three sessions at which an animal is deemed trained
        subj_ses = all_sessions[all_sessions['subject_nickname'] == nickname]
        subj_ses.reset_index()
        trained = ((subj_ses['training_status'] == 'trained_1a')
                   | (subj_ses['training_status'] == 'trained_1b'))
        first_trained = next((w for w, j in enumerate(trained) if j), None)
        ses_select = subj_ses[
                first_trained-days_from_trained[0]+1:first_trained+days_from_trained[1]+1]
        sessions = pd.concat([sessions, ses_select])

    return sessions


def seaborn_style():
    """
    Set seaborn style for plotting figures
    """
    sns.set(style="ticks", context="paper", font_scale=1.2)


def save_csv_subject_query(path):
    """
    Save a csv file with all subjects of the query

    Parameters
    ----------
    path: string with path to where to save file
    """

    subjects = query_subjects(as_dataframe=True)
    subjects.to_csv(os.path.join(path, 'subjects.csv'))
