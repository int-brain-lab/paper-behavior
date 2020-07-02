#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 24 22:32:12 2020
@author: alex
"""
import datajoint as dj

dj.config['database.host'] = 'datajoint.internationalbrainlab.org'
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from os.path import join
import seaborn as sns
from paper_behavior_functions import (seaborn_style, query_sessions_around_criterion, seaborn_style,
                                      institution_map, group_colors, figpath, EXAMPLE_MOUSE,
                                      FIGURE_HEIGHT, FIGURE_WIDTH)
from dj_tools import dj2pandas, fit_psychfunc
from ibl_pipeline import behavior, subject, reference, acquisition
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score, KFold
import os, time
from ibl_pipeline.utils import psychofit as psy
from scipy import stats

seaborn_style()
institution_map, col_names = institution_map()
figpath = figpath()
t = time.time()

##############################################################################
# *******************************Biased Task**********************************#
##############################################################################


# Query sessions biased data
use_sessions, use_days = query_sessions_around_criterion(criterion='biased',
                                                         days_from_criterion=[
                                                             2, 3],
                                                         as_dataframe=False)

# restrict by list of dicts with uuids for these sessions
b = (use_sessions * subject.Subject * subject.SubjectLab * reference.Lab
     * behavior.TrialSet.Trial)

# reduce the size of the fetch
b2 = b.proj('institution_short', 'subject_nickname', 'task_protocol',
            'trial_stim_contrast_left', 'trial_stim_contrast_right',
            'trial_response_choice', 'task_protocol', 'trial_stim_prob_left',
            'trial_feedback_type')
bdat = b2.fetch(order_by=
                'institution_short, subject_nickname, session_start_time, trial_id',
                format='frame').reset_index()
behav_merged = dj2pandas(bdat)

behav_merged['institution_code'] = \
    behav_merged.institution_short.map(institution_map)

# Drop trials with weird contrasts
behav_merged.drop(behav_merged['probabilityLeft']
                  [~behav_merged['signed_contrast'].isin(
        [100, 25, 12.5, 6.25, 0, -6.25, -12.5, -25, -100])].index,
                  inplace=True)

# split the two types of task protocols (remove the pybpod version number
behav_merged['task'] = behav_merged['task_protocol'].str[14:20].copy()

behav = behav_merged.loc[behav_merged['task'] == 'biased'].copy()
behav = behav.reset_index()

for i, nickname in enumerate(np.unique(behav['subject_nickname'])):
    if np.mod(i + 1, 10) == 0:
        print('Loading data of subject %d of %d' % (i + 1, len(
            np.unique(behav['subject_nickname']))))

    # Get the trials of the sessions around criterion
    neutral_n = fit_psychfunc(behav[(behav['subject_nickname'] == nickname)
                                    & (behav['probabilityLeft'] == 50)])
    left_fit = fit_psychfunc(behav[(behav['subject_nickname'] == nickname)
                                   & (behav['probabilityLeft'] == 80)])
    right_fit = fit_psychfunc(behav[(behav['subject_nickname'] == nickname)
                                    & (behav['probabilityLeft'] == 20)])

    behav.loc[behav['subject_nickname'] == nickname, 'threshold'] = \
        neutral_n.loc[0, 'threshold']
    behav.loc[behav['subject_nickname'] == nickname, 'bias_n'] = \
        neutral_n.loc[0, 'bias']
    behav.loc[behav['subject_nickname'] == nickname, 'bias_r'] = \
        right_fit.loc[0, 'bias']
    behav.loc[behav['subject_nickname'] == nickname, 'bias_l'] = \
        left_fit.loc[0, 'bias']

##############################################################################
# *****************************Unbiased Task**********************************#
##############################################################################

# Query sessions traning data
tbehav = behav_merged.loc[behav_merged['task'] == 'traini'].copy()
tbehav.drop(tbehav['probabilityLeft'][~tbehav['probabilityLeft'].isin([50])].index,
            inplace=True)
tbehav = tbehav.reset_index()

for i, nickname in enumerate(np.unique(tbehav['subject_nickname'])):
    if np.mod(i + 1, 10) == 0:
        print('Loading data of subject %d of %d' % (i + 1, len(
            np.unique(tbehav['subject_nickname']))))

    # Get the trials of the sessions around criterion
    trials = tbehav.loc[tbehav['subject_nickname'] == nickname].copy()

    fit_df = dj2pandas(trials.copy())
    fit_result = fit_psychfunc(fit_df)
    tbehav.loc[tbehav['subject_nickname'] == nickname, 'threshold'] = \
        fit_result.loc[0, 'threshold']

t2 = time.time()
print('Elapsed time: %fs'%(t2-t))

##############################################################################
# ***********************************Plot*************************************#
##############################################################################

fig, ax = plt.subplots(1, 1, figsize=(FIGURE_WIDTH / 4, FIGURE_HEIGHT))

selection = behav[behav['subject_nickname'].isin(tbehav['subject_nickname'])]
selection = selection.groupby(['subject_nickname']).mean()
selection['institution'] = [behav.loc[behav['subject_nickname'] == mouse,
                                      'institution_code'].unique()[0] for mouse in selection.index]
selection_t = tbehav.groupby(['subject_nickname']).mean()
sns.regplot(selection_t['threshold'], selection['bias_r'] - selection['bias_l'],
            color='k', scatter=False)
sns.scatterplot(selection_t['threshold'], selection['bias_r'] - selection['bias_l'],
                hue=selection['institution'], palette=group_colors())
ax.set_ylabel('$\Delta$ Rightward choices (%)\nin full task')
ax.get_legend().set_visible(False)
ax.set_xlabel('Visual threshold\n in basic task')

sns.despine(trim=True)
plt.tight_layout()
fig.savefig(os.path.join(figpath, "figure4c_correlation.pdf"))
fig.savefig(os.path.join(figpath, "figure4c_correlation.png"), dpi=300)

dbias = pd.DataFrame()
dbias['bias'] = selection['bias_r'] - selection['bias_l']
dbias['t_threshold'] = selection_t['threshold']
dbias.dropna(inplace=True)
stats.spearmanr(dbias['t_threshold'], dbias['bias'])
stats.pearsonr(dbias['t_threshold'], dbias['bias'])
sns.despine(trim=True)