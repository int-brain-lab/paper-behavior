# -*- coding: utf-8 -*-
"""
Created on Tue Sep  3 10:15:42 2019

Functions for behavioral paper analysis

@author: guido
"""

from ibl_pipeline import subject, acquisition
import seaborn as sns
from os.path import join
from ibl_pipeline.analyses import behavior as behavior_analysis


def query_subjects(as_dataframe=False):
    """
    Query all mice for analysis of behavioral data

    Parameters
    ----------
    as_dataframe: boolean if true returns a pandas dataframe (default is False)
    """

    # Query all subjects with project ibl_neuropixel_brainwide_01 and get the date at which
    # they were flagged as trained_1a
    subj_query = (subject.Subject * subject.SubjectLab * subject.SubjectProject
                  & 'subject_project = "ibl_neuropixel_brainwide_01"').aggr(
                          (acquisition.Session * behavior_analysis.SessionTrainingStatus())
                          & 'training_status="trained_1a"',
                          'subject_nickname', 'sex', 'subject_birth_date', 'lab_name',
                          date_trained='min(date(session_start_time))')

    # Select subjects that reached trained_1a criterium before September 30th
    if as_dataframe is True:
        subjects = (subj_query & 'date_trained < "2019-09-30"').fetch(format='frame')
        subjects = subjects.sort_values(by=['lab_name']).reset_index()
    else:
        subjects = (subj_query & 'date_trained < "2019-09-30"')
    return subjects


def query_sessions(protocol='all', training_status='all', stable=False, as_dataframe=False):
    """
    Query all sessions for analysis of behavioral data

    Parameters
    ----------
    protocol:        string with the following possibilities
                     'all': all sessions (default)
                     'biased': biasedChoiceWorld sessions
                     'training': trainingChoiceWorld sessions
    training_status: string with the following possibilities
                     'all' (default), 'trained_1a', 'trained_1b', 'ready for ephys'
    stable:          boolean if True only return sessions with stable hardware
                     sessions after July 10, 2019 (default is False)
    as_dataframe:    boolean if True returns a pandas dataframe (default is False)
    """

    use_subjects = query_subjects().proj('subject_uuid')
    if protocol == 'all':
        sessions = (acquisition.Session * subject.Subject * subject.SubjectLab * use_subjects
                    & 'task_protocol LIKE "%training%" OR task_protocol LIKE "%biased%"').proj(
                            'session_uuid', 'lab_name', 'subject_nickname', 'task_protocol')
    elif protocol == 'biased' or protocol == 'training':
        sessions = (acquisition.Session * subject.Subject * subject.SubjectLab * use_subjects
                    & 'task_protocol LIKE "%' + protocol + '%"').proj(
                            'session_uuid', 'lab_name', 'subject_nickname', 'task_protocol')
    if training_status != 'all':
        sessions = sessions * (behavior_analysis.SessionTrainingStatus()
                               & 'training_status="%s"' % training_status)
    if stable is True:
        sessions = sessions & 'date(session_start_time) > "2019-06-10"'
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
    subjects.to_csv(join(path, 'subjects.csv'))
