#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 2020-07-20
@author: Anne Urai
"""

import datajoint as dj
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import patsy # to build design matrix
from sklearn.base import BaseEstimator, RegressorMixin
import statsmodels.formula.api as smf
import statsmodels.api as sm
from sklearn.model_selection import cross_val_score

from ibl_pipeline import behavior, subject, reference
from dj_tools import dj2pandas, fit_psychfunc, plot_psychometric
from paper_behavior_functions import (query_sessions_around_criterion,
                                      seaborn_style, institution_map,
                                      group_colors, figpath, EXAMPLE_MOUSE,
                                      FIGURE_WIDTH, FIGURE_HEIGHT)

# Load some things from paper_behavior_functions
figpath = figpath()
seaborn_style()
institution_map, col_names = institution_map()
pal = group_colors()
cmap = sns.diverging_palette(20, 220, n=3, center="dark")

# ========================================== #
#%% 1. LOAD DATA - just from example mouse
# ========================================== #

# Query sessions: before and after full task was first introduced
use_sessions, _ = query_sessions_around_criterion(criterion='biased',
                                                  days_from_criterion=[2, 3],
                                                  as_dataframe=False,
                                                  force_cutoff=True)
use_sessions = (subject.Subject & 'subject_nickname = "%s"' % EXAMPLE_MOUSE) * use_sessions

trial_fields = ('trial_stim_contrast_left', 'trial_stim_contrast_right',
                'trial_response_time', 'trial_stim_prob_left',
                'trial_feedback_type', 'trial_stim_on_time', 'trial_response_choice')

# query trial data for sessions and subject name and lab info
trials = use_sessions.proj('task_protocol') * behavior.TrialSet.Trial.proj(*trial_fields)

# only grab the example mouse
subject_info = (subject.Subject) * \
               (subject.SubjectLab * reference.Lab).proj('institution_short')

# Fetch, join and sort data as a pandas DataFrame
behav = dj2pandas(trials.fetch(format='frame')
                         .join(subject_info.fetch(format='frame'))
                         .sort_values(by=['institution_short', 'subject_nickname',
                                          'session_start_time', 'trial_id'])
                         .reset_index())
# split the two types of task protocols (remove the pybpod version number)
behav['task'] = behav['task_protocol'].str[14:20].copy()

# RECODE SOME THINGS JUST FOR PATSY
behav['contrast'] = np.abs(behav.signed_contrast)
behav['stimulus_side'] = np.sign(behav.signed_contrast)
behav['block_id'] = behav['probabilityLeft'].map({80:-1, 50:0, 20:1})

# ========================================== #
#%% 2. DEFINE THE GLM
# ========================================== #

# DEFINE THE MODEL
def fit_glm(behav, prior_blocks=False):

    # use patsy to easily build design matrix
    if not prior_blocks:
        endog, exog = patsy.dmatrices('choice ~ 1 + stimulus_side:C(contrast, Treatment)'
                                      '+ previous_choice:C(previous_outcome)',
                               data=behav, return_type='dataframe')
    else:
        endog, exog = patsy.dmatrices('choice ~ 1 + stimulus_side:C(contrast, Treatment)'
                                      '+ previous_choice:C(previous_outcome) '
                                      '+ block_id',
                               data=behav, return_type='dataframe')

    # remove the one column (with 0 contrast) that has no variance
    if 'stimulus_side:C(contrast, Treatment)[0.0]' in exog.columns:
        exog.drop(columns=['stimulus_side:C(contrast, Treatment)[0.0]'], inplace=True)

    # recode choices for logistic regression
    endog['choice'] = endog['choice'].map({-1:0, 1:1})

    # rename columns
    exog.rename(columns={'Intercept': 'bias',
              'stimulus_side:C(contrast, Treatment)[6.25]': '6.25',
             'stimulus_side:C(contrast, Treatment)[12.5]': '12.5',
             'stimulus_side:C(contrast, Treatment)[25.0]': '25',
             'stimulus_side:C(contrast, Treatment)[50.0]': '50',
             'stimulus_side:C(contrast, Treatment)[100.0]': '100',
             'previous_choice:C(previous_outcome)[-1.0]': 'unrewarded',
             'previous_choice:C(previous_outcome)[1.0]': 'rewarded'},
             inplace=True)

    # NOW FIT THIS WITH STATSMODELS - ignore NaN choices
    logit_model = sm.Logit(endog, exog, missing='drop')
    res = logit_model.fit_regularized(disp=False)  # run silently

    # what do we want to keep?
    params = pd.DataFrame(res.params).T

    # NOW SIMULATE THIS MODEL MANY TIMES!
    cval = cross_val_score(res, endog, exog)

    return params, cval # wide df

# ========================================== #
#%% 3. FIT
# ========================================== #

print('fitting GLM to BASIC task...')
params_basic, simulation_basic = fit_glm(behav.loc[behav.task == 'traini', :], prior_blocks=False)

print('fitting GLM to FULL task...')
params_full, simulation_full = fit_glm(behav.loc[behav.task == 'biased', :], prior_blocks=True)

# ========================================== #
#%% 4. PLOT PSYCHOMETRIC FUNCTIONS
# ========================================== #

# BASIC TASK
plt.close('all')
fig = sns.FacetGrid(behav.loc[behav.task == 'traini', :],
                    sharex=True, sharey=True,
                    height=FIGURE_HEIGHT, aspect=(FIGURE_WIDTH/4)/FIGURE_HEIGHT)
fig.map(plot_psychometric, "signed_contrast", "choice_right", "session_start_time",
        color='k', linewidth=0)
fig.set_axis_labels('', 'Rightward choices (%)')
fig.despine(trim=True)
fig.savefig(os.path.join(figpath, "figure5b_basic_psychfunc.pdf"))

# FULL TASK
fig = sns.FacetGrid(behav.loc[behav.task == 'biased', :],
                    hue="probabilityLeft", palette=cmap,
                    sharex=True, sharey=True,
                    height=FIGURE_HEIGHT, aspect=(FIGURE_WIDTH/4)/FIGURE_HEIGHT)
fig.map(plot_psychometric, "signed_contrast", "choice_right", "session_start_time",
        linewidth=0)
fig.set_axis_labels('Contrast (%)', 'Rightward choices (%)')
fig.despine(trim=True)
fig.savefig(os.path.join(figpath, "figure5b_full_psychfunc.pdf"))

