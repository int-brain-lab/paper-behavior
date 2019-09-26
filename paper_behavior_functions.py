# -*- coding: utf-8 -*-
"""
Created on Tue Sep  3 10:15:42 2019

Functions for behavioral paper analysis

@author: guido
"""

from ibl_pipeline import subject, acquisition, reference
import seaborn as sns
import os
from ibl_pipeline.analyses import behavior as behavior_analysis


def figpath():
    # Retrieve absolute path of paper-behavior dir
    repo_dir = os.path.dirname(os.path.realpath(__file__))
    # Make figure directory
    fig_dir = os.path.join(repo_dir, 'exported_figs')
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


def query_sessions(stable=False, days_from_trained=0, as_dataframe=False):
    """
    Query all sessions for analysis of behavioral data

    Parameters
    ----------
    stable:          boolean if True only return sessions with stable hardware, which means
                     sessions after July 10, 2019 (default is False)
    as_dataframe:    boolean if True returns a pandas dataframe (default is False)
    days_from_trained: which days? counting from date_trained
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
