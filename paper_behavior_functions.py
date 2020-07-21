# -*- coding: utf-8 -*-
"""
General functions and queries for the analysis of behavioral data from the IBL task

Guido Meijer, Anne Urai, Alejandro Pan Vazquez & Miles Wells
16 Jan 2020
"""

import seaborn as sns
import matplotlib
import os
import numpy as np
import datajoint as dj
# from IPython import embed as shell  # for debugging

# Some constants
QUERY = True  # Whether to query data through DataJoint (True) or use downloaded csv files (False)
EXAMPLE_MOUSE = 'KS014'  # Mouse nickname used as an example
CUTOFF_DATE = '2020-03-23'  # Date after which sessions are excluded, previously 30th Nov
STABLE_HW_DATE = '2019-06-10'  # Date after which hardware was deemed stable

# LAYOUT
FIGURE_HEIGHT = 2  # inch
FIGURE_WIDTH = 8  # inch

# EXCLUDED SESSIONS 
EXCLUDED_SESSIONS = ['a9fb578a-9d7d-42b4-8dbc-3b419ce9f424']  # Session UUID


def group_colors():
    return sns.color_palette("Dark2", 7)


def institution_map():
    institution_map = {'UCL': 'Lab 1', 'CCU': 'Lab 2', 'CSHL': 'Lab 3', 'NYU': 'Lab 4',
                       'Princeton': 'Lab 5', 'SWC': 'Lab 6', 'Berkeley': 'Lab 7'}
    col_names = ['Lab 1', 'Lab 2', 'Lab 3', 'Lab 4', 'Lab 5', 'Lab 6', 'Lab 7', 'All labs']

    return institution_map, col_names


def seaborn_style():
    """
    Set seaborn style for plotting figures
    """
    sns.set(style="ticks", context="paper",
            font="Helvetica",
            rc={"font.size": 9,
                "axes.titlesize": 9,
                "axes.labelsize": 9,
                "lines.linewidth": 1,
                "xtick.labelsize": 7,
                "ytick.labelsize": 7,
                "savefig.transparent": True,
                "xtick.major.size": 2.5,
                "ytick.major.size": 2.5,
                "xtick.minor.size": 2,
                "ytick.minor.size": 2,
                })
    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['ps.fonttype'] = 42


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


def query_subjects(as_dataframe=False, from_list=False, criterion='trained'):
    """
    Query all mice for analysis of behavioral data

    Parameters
    ----------
    as_dataframe:    boolean if true returns a pandas dataframe (default is False)
    from_list:       loads files from list uuids (array of uuids objects)
    criterion:       what criterion by the 30th of November - trained (a and b), biased, ephys
                     (includes ready4ephysrig, ready4delay and ready4recording).  If None,
                     all mice that completed a training session are returned, with date_trained
                     being the date of their first training session.
    """

    from ibl_pipeline import subject, acquisition, reference
    from ibl_pipeline.analyses import behavior as behavior_analysis

    # Query all subjects with project ibl_neuropixel_brainwide_01 and get the date at which
    # they reached a given training status
    all_subjects = (subject.Subject * subject.SubjectLab * reference.Lab * subject.SubjectProject
                    & 'subject_project = "ibl_neuropixel_brainwide_01"')
    sessions = acquisition.Session * behavior_analysis.SessionTrainingStatus()
    fields = ('subject_nickname', 'sex', 'subject_birth_date', 'institution_short')

    if criterion is None:
        # Find first session of all mice; date_trained = date of first training session
        subj_query = all_subjects.aggr(
            sessions, *fields, date_trained='min(date(session_start_time))')
    else:  # date_trained = date of first session when criterion was reached
        if criterion == 'trained':
            restriction = 'training_status="trained_1a" OR training_status="trained_1b"'
        elif criterion == 'biased':
            restriction = 'task_protocol LIKE "%biased%"'
        elif criterion == 'ephys':
            restriction = 'training_status LIKE "ready%"'
        else:
            raise ValueError('criterion must be "trained", "biased" or "ephys"')
        subj_query = all_subjects.aggr(
            sessions & restriction, *fields, date_trained='min(date(session_start_time))')

    if from_list is True:
        ids = np.load('uuids_trained1.npy', allow_pickle=True)
        subj_query = subj_query & [{'subject_uuid': u_id} for u_id in ids]

    # Select subjects that reached criterion before cutoff date
    subjects = (subj_query & 'date_trained <= "%s"' % CUTOFF_DATE)
    if as_dataframe is True:
        subjects = subjects.fetch(format='frame')
        subjects = subjects.sort_values(by=['lab_name']).reset_index()

    return subjects


def query_sessions(task='all', stable=False, as_dataframe=False,
                   force_cutoff=False, criterion='biased'):
    """
    Query all sessions for analysis of behavioral data

    Parameters
    ----------
    task:            string indicating sessions of which task to return, can be trianing or biased
                     default is all
    stable:          boolean if True only return sessions with stable hardware, which means
                     sessions after particular date (default is False)
    as_dataframe:    boolean if True returns a pandas dataframe (default is False)
    force_cutoff:    whether the animal had to reach the criterion by the 30th of Nov. Only
                     applies to biased and ready for ephys criterion
    criterion:       what criterion by the 30th of November - trained (includes
                     a and b), biased, ready (includes ready4ephysrig, ready4delay and
                     ready4recording)
    """

    from ibl_pipeline import acquisition

    # Query sessions
    if force_cutoff is True:
        use_subjects = query_subjects(criterion=criterion).proj('subject_uuid')
    else:
        use_subjects = query_subjects().proj('subject_uuid')

    # Query all sessions or only training or biased if required
    if task == 'all':
        sessions = acquisition.Session * use_subjects & 'task_protocol NOT LIKE "%habituation%"'
    elif task == 'training':
        sessions = acquisition.Session * use_subjects & 'task_protocol LIKE "%training%"'
    elif task == 'biased':
        sessions = acquisition.Session * use_subjects & 'task_protocol LIKE "%biased%"'
    elif task == 'ephys':
        sessions = acquisition.Session * use_subjects & 'task_protocol LIKE "%ephys%"'
    else:
        raise ValueError('task must be "all", "training" or "biased"')

    # Only use sessions up until the end of December
    sessions = sessions & 'date(session_start_time) <= "%s"' % CUTOFF_DATE
    
    # Exclude weird sessions
    sessions = sessions & dj.Not([{'session_uuid': u_id} for u_id in EXCLUDED_SESSIONS])

    # If required only output sessions with stable hardware
    if stable is True:
        sessions = sessions & 'date(session_start_time) > "%s"' % STABLE_HW_DATE

    # Transform into pandas Dataframe if requested
    if as_dataframe is True:
        sessions = sessions.fetch(
            order_by='institution_short, subject_nickname, session_start_time', format='frame')
        sessions = sessions.reset_index()

    return sessions


def query_sessions_around_criterion(criterion='trained', days_from_criterion=(2, 0),
                                    as_dataframe=False, force_cutoff=False):
    """
    Query all sessions for analysis of behavioral data

    Parameters
    ----------
    criterion:              string indicating which criterion to use: trained, biased or ephys
    days_from_criterion:    two-element array which indicates which training days around the day
                            the mouse reached criterium to return, e.g. [3, 2] returns three days
                            before criterium reached up until 2 days after (default: [2, 0])
    as_dataframe:           return sessions as a pandas dataframe
    force_cutoff:           whether the animal had to reach the criterion by the 30th of Nov. Only
                            applies to biased and ready for ephys criterion

    Returns
    ---------
    sessions:               The sessions around the criterion day, works in conjunction with
                            any table that has session_start_time as primary key (such as
                            behavior.TrialSet.Trial)
    days:                   The training days around the criterion day. Can be used in conjunction
                            with tables that have session_date as primary key (such as
                            behavior_analysis.BehavioralSummaryByDate)
    """

    from ibl_pipeline import subject, acquisition
    from ibl_pipeline.analyses import behavior as behavior_analysis

    # Query all included subjects
    if force_cutoff is True:
        use_subjects = query_subjects(criterion=criterion).proj('subject_uuid')
    else:
        use_subjects = query_subjects().proj('subject_uuid')

    # Query per subject the date at which the criterion is reached
    sessions = acquisition.Session * behavior_analysis.SessionTrainingStatus
    if criterion == 'trained':
        restriction = 'training_status="trained_1a" OR training_status="trained_1b"'
    elif criterion == 'biased':
        restriction = 'task_protocol LIKE "%biased%"'
    elif criterion == 'ephys':
        restriction = 'training_status LIKE "ready%"'
    else:
        raise ValueError('criterion must be "trained", "biased" or "ephys"')

    subj_crit = (subject.Subject * use_subjects).aggr(
        sessions & restriction, 'subject_nickname', date_criterion='min(date(session_start_time))')

    # Query the training day at which criterion is reached
    subj_crit_day = (dj.U('subject_uuid', 'day_of_crit')
                     & (behavior_analysis.BehavioralSummaryByDate * subj_crit
                        & 'session_date=date_criterion').proj(day_of_crit='training_day'))

    # Query days around the day at which criterion is reached
    days = (behavior_analysis.BehavioralSummaryByDate * subject.Subject * subj_crit_day
            & ('training_day - day_of_crit between %d and %d'
               % (-days_from_criterion[0], days_from_criterion[1]))).proj(
                   'subject_uuid', 'subject_nickname', 'session_date')

    # Use dates to query sessions
    ses_query = acquisition.Session.aggr(
                            days, from_date='min(session_date)', to_date='max(session_date)')
    
    sessions = (acquisition.Session * ses_query & 'date(session_start_time) >= from_date'
                & 'date(session_start_time) <= to_date')
    
    # Exclude weird sessions
    sessions = sessions & dj.Not([{'session_uuid': u_id} for u_id in EXCLUDED_SESSIONS])

    # Transform to pandas dataframe if necessary
    if as_dataframe is True:
        sessions = sessions.fetch(format='frame').reset_index()
        days = days.fetch(format='frame').reset_index()

    return sessions, days
