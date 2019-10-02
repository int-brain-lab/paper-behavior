"""
PSYCHOMETRIC AND CHRONOMETRIC FUNCTIONS IN BIASED BLOCKS
Anne Urai, CSHL, 2019
"""

from dj_tools import *
import pandas as pd
import numpy as np
import sys
import os
import matplotlib.pyplot as plt
import seaborn as sns
from paper_behavior_functions import *
import datajoint as dj
from IPython import embed as shell  # for debugging
from scipy.special import erf  # for psychometric functions

# import wrappers etc
from ibl_pipeline import reference, subject, action, acquisition, data, behavior
from ibl_pipeline.utils import psychofit as psy
from ibl_pipeline.analyses import behavior as behavioral_analyses

sys.path.insert(0, '../python')

# INITIALIZE A FEW THINGS
sns.set(style="ticks", context="paper", font_scale=1.2)
figpath = figpath()
cmap = sns.diverging_palette(20, 220, n=3, center="dark")
sns.set_palette(cmap)  # palette for water types
pal = sns.color_palette("colorblind", 7)

# ================================= #
# GET DATA FROM TRAINED ANIMALS
# ================================= #


# TODO: WAIT FOR SHAN TO ADD training_day  AND COMPLETE THE QUERY FOR THE RIGHT SESSIONS
use_sessions = query_sessions_around_ephys(days_from_trained=[3, 0])
# restrict by list of dicts with uuids for these sessions
b = acquisition.Session * subject.Subject * subject.SubjectLab * reference.Lab * \
    behavior.TrialSet & use_sessions[['session_uuid']].to_dict(orient='records') * \
    behavioral_analyses.BehavioralSummaryByDate.PsychResults         
bdat = b.fetch(order_by='institution_short, subject_nickname, session_start_time, trial_id', format='frame').reset_index()
behav = dj2pandas(bdat)
assert(~behav.empty)
print(behav.describe())



shell()
# TODO: WAIT FOR SHAN TO ADD training_day  AND COMPLETE THE QUERY FOR THE RIGHT SESSIONS
use_sessions = query_sessions()
sessions = use_sessions & (acquisition.Session & 'task_protocol LIKE %biased%') \
        * behavioral_analyses.BehavioralSummaryByDate.PsychResults

use_sessions = query_sessions(as_dataframe=True)
# restrict by list of dicts with uuids for these sessions
b = (acquisition.Session & 'task_protocol LIKE %biased%')  * subject.Subject * subject.SubjectLab * reference.Lab * \
    use_sessions[['session_uuid']].to_dict(orient='records')  

behavioral_analyses.BehavioralSummaryByDate.PsychResults 

    * behavioral_analyses.SessionTrainingStatus & 'training_status LIKE %ephys' \
behav = sessions.fetch(order_by='institution_short, subject_nickname, session_start_time', format='frame')

shell()