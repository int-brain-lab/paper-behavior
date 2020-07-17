#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 12 16:49:07 2020

@author: Anne Urai
"""
import datajoint as dj
import pandas as pd
import numpy as np
from paper_behavior_functions import (query_sessions_around_criterion,
                                      EXAMPLE_MOUSE, institution_map)
from dj_tools import dj2pandas, fit_psychfunc
from ibl_pipeline import behavior, subject, reference
import statsmodels.api as sm
import patsy # to build design matrix
import os
from tqdm.auto import tqdm
# progress bar
tqdm.pandas(desc="model fitting")

# whether to query data from DataJoint (True), or load from disk (False)
query = True
institution_map, col_names = institution_map()

# ========================================== #
#%% 1. LOAD DATA
# ========================================== #

# Query sessions: before and after full task was first introduced
if query is True:
    use_sessions, _ = query_sessions_around_criterion(criterion='biased',
                                                      days_from_criterion=[2, 3],
                                                      as_dataframe=False,
                                                      force_cutoff=True)

    trial_fields = ('trial_stim_contrast_left', 'trial_stim_contrast_right',
                    'trial_response_time', 'trial_stim_prob_left',
                    'trial_feedback_type', 'trial_stim_on_time', 'trial_response_choice')

    # query trial data for sessions and subject name and lab info
    trials = use_sessions.proj('task_protocol') * behavior.TrialSet.Trial.proj(*trial_fields)
    subject_info = subject.Subject.proj('subject_nickname') * \
                   (subject.SubjectLab * reference.Lab).proj('institution_short')

    # Fetch, join and sort data as a pandas DataFrame
    behav = dj2pandas(trials.fetch(format='frame')
                             .join(subject_info.fetch(format='frame'))
                             .sort_values(by=['institution_short', 'subject_nickname',
                                              'session_start_time', 'trial_id'])
                             .reset_index())
    behav['institution_code'] = behav.institution_short.map(institution_map)
    # split the two types of task protocols (remove the pybpod version number)
    behav['task'] = behav['task_protocol'].str[14:20].copy()

    # RECODE SOME THINGS JUST FOR PATSY
    behav['contrast'] = np.abs(behav.signed_contrast)
    behav['stimulus_side'] = np.sign(behav.signed_contrast)
    behav['block_id'] = behav['probabilityLeft'].map({80:-1, 50:0, 20:1})

else: # load from disk
    behav = pd.read_csv(os.path.join('data', 'Fig5.csv'))

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

    # NOW FIT THIS WITH STATSMODELS
    logit_model = sm.Logit(endog, exog, missing='drop')
    res = logit_model.fit_regularized(disp=False) # run silently

    # what do we want to keep?
    params = pd.DataFrame(res.params).T
    params['pseudo_rsq'] = res.prsquared # https://www.statsmodels.org/stable/generated/statsmodels.discrete.discrete_model.LogitResults.prsquared.html?highlight=pseudo

    return params # wide df

# ========================================== #
#%% 3. FIT FOR EACH MOUSE
# ========================================== #

print('fitting GLM to BASIC task...')
params_basic = behav.loc[behav.task == 'traini', :].groupby(
    ['institution_code', 'subject_nickname']).progress_apply(fit_glm,
                                                     prior_blocks=False).reset_index()

print('fitting GLM to FULL task...')
params_full = behav.loc[behav.task == 'biased', :].groupby(
    ['institution_code', 'subject_nickname']).progress_apply(fit_glm,
                                                     prior_blocks=True).reset_index()

# ========================================== #
# SAVE FOR NEXT TIME
# ========================================== #

params_basic.to_csv('./model_results/params_basic.csv')
params_full.to_csv('./model_results/params_full.csv')

