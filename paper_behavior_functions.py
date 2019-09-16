# -*- coding: utf-8 -*-
"""
Created on Tue Sep  3 10:15:42 2019

Functions for behavioral paper analysis

@author: guido
"""

from ibl_pipeline import subject, acquisition
import seaborn as sns
from ibl_pipeline.analyses import behavior as behavior_analysis


def query_subjects(as_dataframe=False):
    """
    Query all mice for analysis of behavioral data

    Parameters
    ----------
    as_dataframe: boolean if true returns a pandas dataframe (default is False)
    """

    if as_dataframe is True:
        subjects = (subject.Subject * subject.SubjectLab * subject.SubjectProject
                    & 'subject_project = "ibl_neuropixel_brainwide_01"').proj(
                            'subject_nickname', 'sex', 'subject_birth_date', 'lab_name').fetch(
                                    format='frame')
        subjects = subjects.reset_index()
    else:
        subjects = (subject.Subject * subject.SubjectLab * subject.SubjectProject
                    & 'subject_project = "ibl_neuropixel_brainwide_01"').proj(
                            'subject_nickname', 'sex', 'subject_birth_date', 'lab_name')
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
                     'all' (default), 'trained', 'ready for ephys'
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
