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
import statsmodels.api as sm
from ibl_pipeline import behavior, subject, reference
from paper_behavior_functions import (query_sessions_around_criterion,
                                      seaborn_style, institution_map,
                                      group_colors, figpath, EXAMPLE_MOUSE,
                                      FIGURE_WIDTH, FIGURE_HEIGHT,
                                      dj2pandas, fit_psychfunc, plot_psychometric)

# Load some things from paper_behavior_functions
figpath = figpath()
seaborn_style()
institution_map, col_names = institution_map()
pal = group_colors()
#cmap = sns.diverging_palette(20, 220, n=3, center="dark")
cmap = sns.color_palette([[0.8984375,0.37890625,0.00390625],
                          [0.3, 0.3, 0.3], [0.3671875,0.234375,0.59765625]])

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
def fit_glm(behav, prior_blocks=False, n_sim=10000):

    # drop trials with contrast-level 50, only rarely present (should not be its own regressor)
    behav = behav[np.abs(behav.signed_contrast) != 50]

    # use patsy to easily build design matrix
    if not prior_blocks:
        behav = behav.dropna(subset=['trial_feedback_type', 'choice',
                                  'previous_choice', 'previous_outcome']).reset_index()
        endog, exog = patsy.dmatrices('choice ~ 1 + stimulus_side:C(contrast, Treatment)'
                                      '+ previous_choice:C(previous_outcome)',
                               data=behav, return_type='dataframe')
    else:
        behav = behav.dropna(subset=['trial_feedback_type', 'choice',
                                  'previous_choice', 'previous_outcome', 'block_id']).reset_index()
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
    logit_model = sm.Logit(endog, exog)
    res = logit_model.fit_regularized(disp=False)  # run silently

    # what do we want to keep?
    params = pd.DataFrame(res.params).T

    # USE INVERSE HESSIAN TO CONSTRUCT MULTIVARIATE GAUSSIAN
    cov = -np.linalg.inv(logit_model.hessian(res.params))
    samples = np.random.multivariate_normal(res.params, cov, n_sim)

    # sanity check: the mean of those samples should not be too different from the params
    assert(np.allclose(params, np.mean(samples, axis=0), atol=0.1))

    # NOW SIMULATE THE MODEL X TIMES
    simulated_choices = []
    for n in range(n_sim):
        # plug sampled parameters into the model - predict sequence of choices
        z = np.dot(exog, samples[n])

        # then compute the mean choice fractions at each contrast, save and append
        behav['simulated_choice'] = 1 / (1 + np.exp(-z))
        if not prior_blocks:
            simulated_choices.append(behav.groupby(['signed_contrast'])['simulated_choice'].mean().values)
        else: # split by probabilityLeft block
            gr = behav.groupby(['probabilityLeft', 'signed_contrast'])['simulated_choice'].mean().reset_index()
            simulated_choices.append([gr.loc[gr.probabilityLeft == 20, 'simulated_choice'].values,
                                      gr.loc[gr.probabilityLeft == 50, 'simulated_choice'].values,
                                      gr.loc[gr.probabilityLeft == 80, 'simulated_choice'].values])

    return params, simulated_choices  # wide df

# ========================================== #
#%% 3. FIT
# ========================================== #

print('fitting GLM to BASIC task...')
params_basic, simulation_basic = fit_glm(behav.loc[behav.task == 'traini', :],
                                         prior_blocks=False)

print('fitting GLM to FULL task...')
params_full, simulation_full = fit_glm(behav.loc[behav.task == 'biased', :],
                                       prior_blocks=True)

# ========================================== #
#%% 4. PLOT PSYCHOMETRIC FUNCTIONS
# ========================================== #

# for plotting, replace 100 with -35
behav['signed_contrast'] = behav['signed_contrast'].replace(-100, -35)
behav['signed_contrast'] = behav['signed_contrast'].replace(100, 35)

# BASIC TASK
plt.close('all')
# prep the figure with psychometric layout
fig = sns.FacetGrid(behav.loc[behav.task == 'traini', :],
                    sharex=True, sharey=True,
                    height=FIGURE_HEIGHT, aspect=(FIGURE_WIDTH/4)/FIGURE_HEIGHT)
fig.map(plot_psychometric, "signed_contrast", "choice_right", "subject_nickname",
        color='k', linewidth=0) # this will be empty, hack
# now plot the datapoints, no errorbars
sns.lineplot(data=behav.loc[behav.task == 'traini', :],
             x='signed_contrast', y='choice2', marker='o', err_style='bars',
             color='k', linewidth=0, ci=95, ax=fig.ax)# overlay the simulated
# confidence intervals from the model - shaded regions
fig.ax.fill_between(sorted(behav.signed_contrast.unique()),
                    np.quantile(np.array(simulation_basic), q=0.025, axis=0),
                    np.quantile(np.array(simulation_basic), q=0.975, axis=0),
                    alpha=0.5, facecolor='k')
fig.set_axis_labels(' ', 'Rightward choices (%)')
fig.despine(trim=True)
fig.savefig(os.path.join(figpath, "figure5b_basic_psychfunc.pdf"))

# FULL TASK
plt.close('all')
fig = sns.FacetGrid(behav.loc[behav.task == 'biased', :],
                    hue="probabilityLeft", palette=cmap,
                    sharex=True, sharey=True,
                    height=FIGURE_HEIGHT, aspect=(FIGURE_WIDTH/4)/FIGURE_HEIGHT)
fig.map(plot_psychometric, "signed_contrast", "choice_right", "subject_nickname", linewidth=0) # just for axis layout,
# hack
# now plot the datapoints, no errorbars
sns.lineplot(data=behav.loc[behav.task == 'biased', :],
             x='signed_contrast', y='choice2', marker='o', err_style='bars',
             hue='probabilityLeft', palette=cmap, linewidth=0, ci=95, ax=fig.ax, legend=None)# overlay the simulated
# confidence intervals from the model - shaded regions
for cidx, c in enumerate(cmap):
    simulation_full_perblock = [sim[cidx] for sim in simulation_full] # grab what we need, not super elegant
    fig.ax.fill_between(sorted(behav.signed_contrast.unique()),
                    np.quantile(np.array(simulation_full_perblock), q=0.025, axis=0),
                    np.quantile(np.array(simulation_full_perblock), q=0.975, axis=0),
                    alpha=0.5, facecolor=cmap[cidx])

fig.ax.annotate('20:80', xy=(-5, 0.6), xytext=(-25, 0.8), color=cmap[0], fontsize=7)
fig.ax.annotate('80:20', xy=(5, 0.4), xytext=(13, 0.18), color=cmap[2], fontsize=7)

fig.set_axis_labels('Contrast (%)', 'Rightward choices (%)')
fig.despine(trim=True)
fig.savefig(os.path.join(figpath, "figure5b_full_psychfunc.pdf"))

