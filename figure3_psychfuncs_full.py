"""
Psychometric functions of training mice, within and across labs

@author: Anne Urai
15 January 2020
"""
import seaborn as sns
import os
import matplotlib.pyplot as plt
from paper_behavior_functions import (figpath, seaborn_style, group_colors,
                                      query_sessions_around_criterion, institution_map)
from ibl_pipeline import reference, subject, behavior
from dj_tools import plot_psychometric, dj2pandas, plot_chronometric

# initialize
figpath = figpath()
pal = group_colors()
institution_map, col_names = institution_map()
col_names = col_names[:-1]

# query sessions
use_sessions = query_sessions_around_criterion(criterion='biased', days_from_criterion=[0, 15])[0]
b = (use_sessions * subject.Subject * subject.SubjectLab * reference.Lab * behavior.TrialSet.Trial
     & 'task_protocol LIKE "%biased%"')

# load data into pandas dataframe
b2 = b.proj('institution_short', 'subject_nickname', 'task_protocol',
            'trial_stim_contrast_left', 'trial_stim_contrast_right', 'trial_response_choice',
            'task_protocol', 'trial_stim_prob_left', 'trial_feedback_type',
            'trial_response_time', 'trial_stim_on_time')
bdat = b2.fetch(order_by='institution_short, subject_nickname, session_start_time, trial_id',
                format='frame').reset_index()
behav = dj2pandas(bdat)
behav['institution_code'] = behav.institution_short.map(institution_map)

# exclude contrasts that were part of a pilot with a different contrast set
behav = behav[((behav['signed_contrast'] != -8) & (behav['signed_contrast'] != -4)
               & (behav['signed_contrast'] != 4) & (behav['signed_contrast'] != 8))]

# select only 50/50 block trials
left = behav[behav['probabilityLeft'] == 80]
right = behav[behav['probabilityLeft'] == 20]


