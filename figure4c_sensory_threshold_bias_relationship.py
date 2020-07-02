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
import os, time
from ibl_pipeline.utils import psychofit as psy
from scipy import stats

# whether to query data from DataJoint (True), or load from disk (False)
query = True

# initialize
seaborn_style()
institution_map, col_names = institution_map()
figpath = figpath()

# %% QUERY

if query is True:
    # Query sessions biased data 
    use_sessions, _ = query_sessions_around_criterion(criterion='biased',
                                                      days_from_criterion=[2, 3])
    
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
    behav = dj2pandas(bdat)
    behav['institution_code'] = behav.institution_short.map(institution_map)
    #
    # # exclude contrasts that were part of a pilot with a different contrast set
    # behav_merged = behav_merged[((behav_merged['signed_contrast'] != -8)
    #                              & (behav_merged['signed_contrast'] != -4)
    #                              & (behav_merged['signed_contrast'] != 4)
    #                              & (behav_merged['signed_contrast'] != 8))]

else:
    behav = pd.read_csv(join('data', 'Fig4c_and_Fig5.csv'))

# split the two types of task protocols (remove the pybpod version number
behav['task'] = behav['task_protocol'].str[14:20].copy()
behav.loc[behav.task == 'traini', 'probabilityLeft'] = 50
behav_all = behav.copy()

# %% fit psychfuncs per task and leftProb
print('fitting psychometric functions...')
t = time.time()
pars = behav.groupby(['institution_code', 'subject_nickname', 'task',
                      'probabilityLeft']).apply(fit_psychfunc).reset_index()
# separate out
pars_training = pars.loc[pars['task']=='traini'].copy()
pars_biased = pars.loc[pars['task']=='biased'].copy()

# then, take the difference between the two conditions
pars_biased2 = pd.pivot_table(pars_biased, values='bias',
                       index=['institution_code', 'subject_nickname', 'task'],
                       columns=['probabilityLeft']).reset_index()
pars_biased2['bias_shift'] = pars_biased2[20] - pars_biased2[80]

# %% in the biased task, compute the shift between two conditions
# first, transform the psychometric pars in to % rightward at 0% contrast
def pars2prob(pars, xvec=0):
    yvec = psy.erf_psycho_2gammas([pars.bias.item(),
                               pars.threshold.item(),
                               pars.lapselow.item(),
                               pars.lapsehigh.item()], xvec)
    return yvec

print('transforming parameters to % rightward choices at contrast 0...')
choice_right = pars_biased.groupby(['institution_code', 'subject_nickname',
                      'probabilityLeft']).apply(pars2prob).reset_index()

# then, take the difference between the two conditions
choice_right2 = pd.pivot_table(choice_right, values=0,
                       index=['institution_code', 'subject_nickname'],
                       columns=['probabilityLeft']).reset_index()
choice_right2['choiceprob_shift'] = (choice_right2[20] - choice_right2[80]) * 100

# %% merge dfs back together
behav_merged = pd.merge(choice_right2[['institution_code', 'subject_nickname', 'choiceprob_shift']],
                        pars_biased2[['institution_code', 'subject_nickname', 'bias_shift']],
                        on=['institution_code', 'subject_nickname'])
behav_merged = pd.merge(behav_merged,
                        pars_training[['institution_code', 'subject_nickname', 'threshold']],
                        on=['institution_code', 'subject_nickname'])
t2 = time.time()
print('Elapsed time: %fs'%(t2-t))

# # %% do Alex' way for timing test
# t = time.time()
# behav = behav_all.loc[behav_all['task']=='biased'].copy()
# behav = behav.reset_index()
# xvec = behav.signed_contrast.unique()
#
# for i, nickname in enumerate(np.unique(behav['subject_nickname'])):
#         if np.mod(i+1, 10) == 0:
#             print('Loading data of subject %d of %d' % (i+1, len(
#                     np.unique(behav['subject_nickname']))))
#
#         # Get the trials of the sessions around criterion
#         neutral_n = fit_psychfunc(behav[(behav['subject_nickname'] == nickname)
#                                    & (behav['probabilityLeft'] == 50)])
#         left_fit = fit_psychfunc(behav[(behav['subject_nickname'] == nickname)
#                                            & (behav['probabilityLeft'] == 80)])
#         right_fit = fit_psychfunc(behav[(behav['subject_nickname'] == nickname)
#                                             & (behav['probabilityLeft'] == 20)])
#
#         behav.loc[behav['subject_nickname'] == nickname, 'threshold'] = \
#                 neutral_n.loc[0, 'threshold']
#         behav.loc[behav['subject_nickname'] == nickname, 'bias_n'] = \
#                 neutral_n.loc[0, 'bias']
#
#         # instaed of the bias term, compute delta rightward choices
#         behav.loc[behav['subject_nickname'] == nickname, 'bias_r'] = \
#             psy.erf_psycho_2gammas([right_fit.bias.item(),
#                                     right_fit.threshold.item(),
#                                     right_fit.lapselow.item(),
#                                     right_fit.lapsehigh.item()], 0)
#         behav.loc[behav['subject_nickname'] == nickname, 'bias_l'] = \
#             psy.erf_psycho_2gammas([left_fit.bias.item(),
#                                     left_fit.threshold.item(),
#                                     left_fit.lapselow.item(),
#                                     left_fit.lapsehigh.item()], 0)
#
# ##############################################################################
# #*****************************Unbiased Task**********************************#
# ##############################################################################
#
# # Query sessions traning data
# tbehav = behav_all.loc[behav_all['task']=='traini'].copy()
# tbehav.drop(tbehav['probabilityLeft'][~tbehav['probabilityLeft'].isin([50])].index,
#         inplace=True)
# tbehav = tbehav.reset_index()
#
# for i, nickname in enumerate(np.unique(tbehav['subject_nickname'])):
#         if np.mod(i+1, 10) == 0:
#             print('Loading data of subject %d of %d' % (i+1, len(
#                     np.unique(tbehav['subject_nickname']))))
#
#         # Get the trials of the sessions around criterion
#         trials = tbehav.loc[tbehav['subject_nickname'] == nickname].copy()
#         fit_df = dj2pandas(trials.copy())
#         fit_result = fit_psychfunc(fit_df)
#         tbehav.loc[tbehav['subject_nickname'] == nickname, 'threshold'] = \
#                     fit_result.loc[0, 'threshold']
#
# #############################################################################
# #***********************************Plot*************************************#
# #############################################################################
#
# # why copy these data again?
# selection = behav[behav['subject_nickname'].isin(tbehav['subject_nickname'])]
# selection = selection.groupby(['subject_nickname']).mean()
# selection['institution'] = [behav.loc[behav['subject_nickname'] == mouse,
#             'institution_code'].unique()[0]for mouse in selection.index]
# selection_t = tbehav.groupby(['subject_nickname']).mean()
#
# t2 = time.time()
# print('Elapsed time: %fs'%(t2-t))

# %% PLOT - delta rightward choices
fig, ax = plt.subplots(1,1,figsize=(FIGURE_WIDTH/4, FIGURE_HEIGHT))
sns.regplot(behav_merged['threshold'], behav_merged['choiceprob_shift'],
            color='k', scatter=False)
sns.scatterplot(behav_merged['threshold'], behav_merged['choiceprob_shift'],
                hue=behav_merged['institution_code'], palette=group_colors())
ax.set_ylabel('$\Delta$ Rightward choices (%)\nin full task')
ax.get_legend().set_visible(False)
ax.set_xlabel('Visual threshold\n in basic task')
sns.despine(trim=True)
plt.tight_layout()
fig.savefig(os.path.join(figpath, "figure4c_correlation_choiceprob.pdf"))
fig.savefig(os.path.join(figpath, "figure4c_correlation_choiceprob.png"), dpi=300)

#  PRINT STATS ON THE CORRELATION
print(stats.spearmanr(behav_merged['threshold'],
                      behav_merged['choiceprob_shift'], nan_policy='omit'))
# print(stats.pearsonr(behav_merged['threshold'], behav_merged['choiceprob_shift']))

# %% PLOT - biasshift from psychfunc
fig, ax = plt.subplots(1,1,figsize=(FIGURE_WIDTH/4, FIGURE_HEIGHT))
sns.regplot(behav_merged['threshold'], behav_merged['bias_shift'],
            color='k', scatter=False)
sns.scatterplot(behav_merged['threshold'], behav_merged['bias_shift'],
                hue=behav_merged['institution_code'], palette=group_colors())
ax.set_ylabel('$\Delta \mu$ in full task')
ax.get_legend().set_visible(False)
ax.set_xlabel('Visual threshold\n in basic task')
sns.despine(trim=True)
plt.tight_layout()
fig.savefig(os.path.join(figpath, "figure4c_correlation_biasshift.pdf"))
fig.savefig(os.path.join(figpath, "figure4c_correlation_biasshift.png"), dpi=300)

#  PRINT STATS ON THE CORRELATION
print(stats.spearmanr(behav_merged['threshold'],
                      behav_merged['bias_shift'], nan_policy='omit'))
# print(stats.pearsonr(behav_merged['threshold'], behav_merged['choiceprob_shift']))
