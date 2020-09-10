#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 2020-07-20
@author: Anne Urai
"""
import datajoint as dj
import pandas as pd
import numpy as np
from paper_behavior_functions import (query_sessions_around_criterion,
                                      EXAMPLE_MOUSE, institution_map,
                                      dj2pandas, fit_psychfunc)
from ibl_pipeline import behavior, subject, reference
import os
from tqdm.auto import tqdm
from sklearn.model_selection import KFold

# for modelling
import patsy # to build design matrix
import statsmodels.api as sm

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
def fit_glm(behav, prior_blocks=False, folds=5):

    # drop trials with contrast-level 50, only rarely present (should not be its own regressor)
    behav = behav[np.abs(behav.signed_contrast) != 50]

    # use patsy to easily build design matrix
    if not prior_blocks:
        endog, exog = patsy.dmatrices('choice ~ 1 + stimulus_side:C(contrast, Treatment)'
                                      '+ previous_choice:C(previous_outcome)',
                               data=behav.dropna(subset=['trial_feedback_type', 'choice',
                                  'previous_choice', 'previous_outcome']).reset_index(),
                                      return_type='dataframe')
    else:
        endog, exog = patsy.dmatrices('choice ~ 1 + stimulus_side:C(contrast, Treatment)'
                                      '+ previous_choice:C(previous_outcome) '
                                      '+ block_id',
                               data=behav.dropna(subset=['trial_feedback_type', 'choice',
                                  'previous_choice', 'previous_outcome', 'block_id']).reset_index(),
                                      return_type='dataframe')

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
    logit_model = sm.Logit(endog, exog)
    res = logit_model.fit_regularized(disp=False) # run silently

    # what do we want to keep?
    params = pd.DataFrame(res.params).T
    params['pseudo_rsq'] = res.prsquared # https://www.statsmodels.org/stable/generated/statsmodels.discrete.discrete_model.LogitResults.prsquared.html?highlight=pseudo
    params['condition_number'] = np.linalg.cond(exog)
    
    # ===================================== #
    # ADD MODEL ACCURACY - cross-validate

    kf = KFold(n_splits=folds, shuffle=True)
    acc = np.array([])
    for train, test in kf.split(endog):
        X_train, X_test, y_train, y_test = exog.loc[train], exog.loc[test], \
                                           endog.loc[train], endog.loc[test]
        # fit again
        logit_model = sm.Logit(y_train, X_train)
        res = logit_model.fit_regularized(disp=False)  # run silently

        # compute the accuracy on held-out data [from Luigi]:
        # suppose you are predicting Pr(Left), let's call it p,
        # the % match is p if the actual choice is left, or 1-p if the actual choice is right
        # if you were to simulate it, in the end you would get these numbers
        y_test['pred'] = res.predict(X_test)
        y_test.loc[y_test['choice'] == 0, 'pred'] = 1 - y_test.loc[y_test['choice'] == 0, 'pred']
        acc = np.append(acc, y_test['pred'].mean())

    # average prediction accuracy over the K folds
    params['accuracy'] = np.mean(acc)

    return params  # wide df


# ========================================== #
#%% 3. FIT FOR EACH MOUSE
# ========================================== #

print('fitting GLM to BASIC task...')
params_basic = behav.loc[behav.task == 'traini', :].groupby(
    ['institution_code', 'subject_nickname']).progress_apply(fit_glm,
                                                     prior_blocks=False).reset_index()
print('The mean condition number for the basic model is', params_basic['condition_number'].mean())
                                                             
print('fitting GLM to FULL task...')
params_full = behav.loc[behav.task == 'biased', :].groupby(
    ['institution_code', 'subject_nickname']).progress_apply(fit_glm,
                                                     prior_blocks=True).reset_index()
print('The mean condition number for the full model is', params_full['condition_number'].mean())

# ========================================== #
# SAVE FOR NEXT TIME
# ========================================== #

params_basic.to_csv('./model_results/params_basic.csv')
params_full.to_csv('./model_results/params_full.csv')

