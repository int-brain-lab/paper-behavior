#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
By AnneU

"""
import datajoint as dj
import pandas as pd
import matplotlib.pyplot as plt
from os.path import join
import seaborn as sns
from paper_behavior_functions import (seaborn_style, query_sessions_around_criterion, seaborn_style, institution_map, group_colors, figpath, EXAMPLE_MOUSE,
                                      FIGURE_HEIGHT, FIGURE_WIDTH)
from dj_tools import dj2pandas, fit_psychfunc
from ibl_pipeline import behavior, subject, reference
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
    b2 = b.proj('institution_short', 'subject_nickname', 'task_protocol', 'session_uuid',
                'trial_stim_contrast_left', 'trial_stim_contrast_right', 'trial_response_choice',
                'task_protocol', 'trial_stim_prob_left', 'trial_feedback_type',
                'trial_response_time', 'trial_stim_on_time')

    # construct pandas dataframe
    bdat = b2.fetch(order_by='institution_short, subject_nickname, session_start_time, trial_id',
                    format='frame').reset_index()
    behav = dj2pandas(bdat)
    behav['institution_code'] = behav.institution_short.map(institution_map)

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
behav_merged.to_csv(os.path.join(figpath, 'correlation_anne.csv'))

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


# %% PLOT - biasshift from psychfunc
fig, ax = plt.subplots(1,1,figsize=(FIGURE_WIDTH/4, FIGURE_HEIGHT))
sns.regplot(behav_merged['choiceprob_shift'], behav_merged['bias_shift'],
            color='k', scatter=False)
sns.scatterplot(behav_merged['choiceprob_shift'], behav_merged['bias_shift'],
                hue=behav_merged['institution_code'], palette=group_colors())
ax.set_ylabel('$\Delta \mu$ in full task')
ax.get_legend().set_visible(False)
ax.set_xlabel('Choice prob shift')
sns.despine(trim=True)
plt.tight_layout()
fig.savefig(os.path.join(figpath, "figure4c_correlation_check.pdf"))

#  PRINT STATS ON THE CORRELATION
print(stats.spearmanr(behav_merged['choiceprob_shift'],
                      behav_merged['bias_shift'], nan_policy='omit'))
# print(stats.pearsonr(behav_merged['threshold'], behav_merged['choiceprob_shift']))


# %% COMPARE THE TWO!
csv_anne = pd.read_csv(os.path.join(figpath, 'correlation_anne.csv'))
csv_alex = pd.read_csv(os.path.join(figpath, 'correlation_alex.csv'))
csv_merged = pd.merge(csv_anne, csv_alex, on=['subject_nickname'])
csv_merged = csv_merged.rename(columns={'threshold':'threshold_anne',
                                        'threshold_y':'threshold_alex',
                                        'bias_shift':'biasshift_anne',
                                        'biasshift':'biasshift_alex',
                                        'choiceprob_shift':'choiceprobshift_anne'})

g = sns.pairplot(csv_merged[['threshold_anne', 'threshold_alex',
                             'biasshift_anne', 'biasshift_alex',
                             'choiceprobshift_anne']], corner=True)
sns.despine(trim=True)
plt.tight_layout()
g.savefig(os.path.join(figpath, "figure4c_correlation_pairplot.pdf"))
