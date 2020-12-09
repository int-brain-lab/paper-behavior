"""
Choice variability across contrasts within each lab and across labs

@author: Anne Urai
2 September 2020
"""

import numpy as np
import seaborn as sns
import os
import pandas as pd
import matplotlib.pyplot as plt
from paper_behavior_functions import (figpath, seaborn_style, group_colors,
                                      query_sessions_around_criterion, institution_map,
                                      FIGURE_HEIGHT, FIGURE_WIDTH, dj2pandas)
# import wrappers etc
from ibl_pipeline import reference, subject, behavior
from sklearn.utils import shuffle

# progress bar
from tqdm.auto import tqdm
tqdm.pandas(desc="computing")

# Initialize
seaborn_style()
figpath = figpath()
pal = group_colors()
institution_map, col_names = institution_map()
col_names = col_names[:-1]
nshuf = 10000


# DEFINE ACROSS-MOUSE CHOICE VARIABILITY
def choice_variability(df):
    # FIRST, GROUP BY CONTRAST AND MOUSE
    choicedat = df.pivot_table(values='choice2', columns='subject_nickname', index='signed_contrast')

    # COMPUTE THE ACROSS-MOUSE VARIANCE FOR EACH CONTRAST
    # AVERAGE VARIANCE ACROSS CONTRASTS
    choice_var = 1 / choicedat.var(axis=1).mean()

    return choice_var


# FULL TASK, ALSO INCORPORATE PROBABILITYLEFT
def choice_variability_full(df):
    # FIRST, GROUP BY CONTRAST AND MOUSE
    choicedat = df.pivot_table(values='choice2',
                               columns='subject_nickname',
                               index=['signed_contrast', 'probabilityLeft'])

    # COMPUTE THE ACROSS-MOUSE VARIANCE FOR EACH CONTRAST
    # AVERAGE VARIANCE ACROSS CONTRASTS
    choice_var = 1 / choicedat.var(axis=1).mean()

    return choice_var


# FULL TASK, ACROSS-LAB VARIABILITY IN BIAS SHIFT
def biasshift_variability_full(df):
    # FIRST, GROUP BY CONTRAST AND MOUSE
    choicedat = df.pivot_table(values='biasshift',
                               columns='subject_nickname',
                               index=['signed_contrast'])

    # COMPUTE THE ACROSS-MOUSE VARIANCE FOR EACH CONTRAST
    # AVERAGE VARIANCE ACROSS CONTRASTS
    choice_var = 1 / choicedat.var(axis=1).mean()

    return choice_var

# ================================= #
# GET DATA FROM TRAINED ANIMALS
# ================================= #

# query sessions - BASIC TASK
use_sessions, _ = query_sessions_around_criterion(criterion='trained',
                                                  days_from_criterion=[0, 2],
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
behav.drop(behav[behav['signed_contrast'].isin([50., -50])].index, inplace=True)
df_basic = behav.groupby(['institution_code', 'subject_nickname', 'signed_contrast']).agg(
    {'choice2': 'mean'}).reset_index().copy()

# query sessions - FULL TASK
use_sessions, _ = query_sessions_around_criterion(criterion='ephys',
                                                  days_from_criterion=[2, 0],
                                                  force_cutoff=True)
use_sessions = use_sessions & 'task_protocol LIKE "%biased%"'  # only get biased sessions

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

# COMPUTE AVERAGE CHOICE BEHAVIOR FOR EACH SJ
df_full = (
    behav
    .groupby(['institution_code', 'probabilityLeft', 'subject_nickname', 'signed_contrast'])
    .agg({'choice2': 'mean'})
    .reset_index()
    .copy()
)

# ================================================================== #
# COMPUTE CHOICE VARIABILITY FOR EACH LAB - BASIC TASK
# ================================================================== #

basic_perlab = df_basic.groupby(['institution_code']).progress_apply(choice_variability).reset_index()
basic_perlab['x'] = 0
basic_perlab['choice_var'] = basic_perlab[0]


# # SAME, BUT ON SHUFFLED DATA
# basic_perlab_shuffled = []
# # make a list of all the institution codes for all subjects (preserving their frequency)
# institution_codes = [gr.institution_code.unique()[0] for _, gr in df_basic.groupby(['subject_nickname'])]
# for s in tqdm(range(nshuf)):
#
#     # use scikit learn to shuffle lab labels
#     shuffled_labs = shuffle(institution_codes)
#     new_df = []
#     # keep all choices of one mouse together - only reassign labs!
#     for idx, g in enumerate(df_basic.groupby(['subject_nickname'])):
#         g[1]['new_lab'] = shuffled_labs[idx]
#         new_df.append(g[1])
#     df_basic = pd.concat(new_df)
#
#     assert (all(df_basic.groupby(['institution_code'])['new_lab'].nunique() > 1))
#     basic_perlab_shuffled.append(df_basic.groupby(['new_lab']).apply(choice_variability).mean())
#
# # PLOT
# f, ax1 = plt.subplots(1, 1, figsize=(FIGURE_WIDTH / 5, FIGURE_HEIGHT))
# sns.swarmplot(data = basic_perlab,
#               x ='x', y='choice_var', hue_order=col_names,
#               hue = 'institution_code',
#               palette = pal, marker='.', ax=ax1, zorder=0)
# ax1.plot(0, basic_perlab['choice_var'].mean(),
#              color='black', linewidth=0, marker='_', markersize=13)
# ax1.get_legend().set_visible(False)
# # then, shuffled distribution next to it
# sns.violinplot(x=np.concatenate((np.zeros(nshuf), np.ones(nshuf))),
#               y=np.concatenate((np.empty((nshuf))*np.nan, basic_perlab_shuffled)),
#                palette=[[1,1,1]], ax=ax1)
# ax1.set(ylabel='Within-lab choice consistency', xlabel='', ylim=[0, 120])
# ax1.set_xticklabels(['Data', 'Shuffle'], ha='center')
# plt.tight_layout()
# sns.despine(trim=True)
# f.savefig(os.path.join(figpath, "across_mouse_var_basic.pdf"))
#
# # WHAT IS THE P-VALUE COMPARED TO THE  NULL DISTRIBUTION?
# pval = np.mean(basic_perlab_shuffled > basic_perlab['choice_var'].mean())
# print('Basic task, choice consistency: p-value = %.4f'%pval)

# ================================================================== #
# COMPUTE CHOICE VARIABILITY FOR EACH LAB - full TASK
# ================================================================== #

full_perlab = df_full.groupby(['institution_code']).progress_apply(choice_variability_full).reset_index()
full_perlab['x'] = 0
full_perlab['choice_var'] = full_perlab[0]

# # SAME, BUT ON SHUFFLED DATA
# full_perlab_shuffled = []
# # make a list of all the institution codes for all subjects (preserving their frequency)
# institution_codes = [gr.institution_code.unique()[0] for _, gr in df_full.groupby(['subject_nickname'])]
# for s in tqdm(range(nshuf)):
#
#     # use scikit learn to shuffle lab labels
#     shuffled_labs = shuffle(institution_codes)
#     new_df = []
#     # keep all choices of one mouse together - only reassign labs!
#     for idx, g in enumerate(df_full.groupby(['subject_nickname'])):
#         g[1]['new_lab'] = shuffled_labs[idx]
#         new_df.append(g[1])
#     df_full = pd.concat(new_df)
#
#     assert (all(df_full.groupby(['institution_code'])['new_lab'].nunique().mean() > 1))
#     full_perlab_shuffled.append(df_full.groupby(['new_lab']).apply(choice_variability_full).mean())
#
#
# # PLOT
# f, ax1 = plt.subplots(1, 1, figsize=(FIGURE_WIDTH / 5, FIGURE_HEIGHT))
# sns.swarmplot(data = full_perlab,
#               x ='x', y='choice_var', hue_order=col_names,
#               hue = 'institution_code',
#               palette = pal, marker='.', ax=ax1, zorder=0)
# ax1.plot(0, full_perlab['choice_var'].mean(),
#              color='black', linewidth=0, marker='_', markersize=13)
# ax1.get_legend().set_visible(False)
# # then, shuffled distribution next to it
# sns.violinplot(x=np.concatenate((np.zeros(nshuf), np.ones(nshuf))),
#               y=np.concatenate((np.empty((nshuf))*np.nan, full_perlab_shuffled)),
#                palette=[[1,1,1]], ax=ax1)
# ax1.set(ylabel=' ', xlabel='', ylim=[0, 120])
# ax1.set_xticklabels(['Data', 'Shuffle'], ha='center')
# plt.tight_layout()
# sns.despine(trim=True)
# f.savefig(os.path.join(figpath, "across_mouse_var_full.pdf"))
#
# # WHAT IS THE P-VALUE COMPARED TO THE  NULL DISTRIBUTION?
# pval = np.mean(full_perlab_shuffled > full_perlab['choice_var'].mean())
# print('Full task, choice consistency: p-value = %.4f'%pval)



# ================================================================== #
# FULL TASK - VARIABILITY IN BIAS SHIFT
# ================================================================== #

# convert choices per probabilityLeft into bias shift
df_full2 = df_full.pivot_table(values='choice2', columns='probabilityLeft',
                               index=['institution_code', 'subject_nickname',
                                      'signed_contrast']).reset_index().copy()
df_full2['biasshift'] = (df_full2[20] - df_full2[80])

# COMPUTE THE BIAS SHIFT PER CONTRAST FOR EACH MOUSE
full_biasshift_perlab = df_full2.groupby(['institution_code']
                                         ).progress_apply(biasshift_variability_full).reset_index()
full_biasshift_perlab['biasshift_var'] = full_biasshift_perlab[0]
full_biasshift_perlab['x'] = 0

# SAME, BUT ON SHUFFLED DATA
full_biasshift_perlab_shuffled = []
# make a list of all the institution codes for all subjects (preserving their frequency)
institution_codes = [gr.institution_code.unique()[0] for _, gr in df_full2.groupby(['subject_nickname'])]
for s in tqdm(range(nshuf)):

    # use scikit learn to shuffle lab labels
    shuffled_labs = shuffle(institution_codes)
    new_df = []
    # keep all choices of one mouse together - only reassign labs!
    for idx, g in enumerate(df_full2.groupby(['subject_nickname'])):
        g[1]['new_lab'] = shuffled_labs[idx]
        new_df.append(g[1])
    df_full2 = pd.concat(new_df)

    assert (all(df_full2.groupby(['institution_code'])['new_lab'].nunique() > 1))
    full_biasshift_perlab_shuffled.append(df_full2.groupby(['new_lab'
                                                            ]).apply(biasshift_variability_full).mean())


# PLOT
f, ax1 = plt.subplots(1, 1, figsize=(FIGURE_WIDTH / 5, FIGURE_HEIGHT))
sns.swarmplot(data = full_biasshift_perlab,
              x ='x', y='biasshift_var', hue_order=col_names,
              hue = 'institution_code',
              palette = pal, marker='.', ax=ax1, zorder=0)
ax1.plot(0, full_biasshift_perlab['biasshift_var'].mean(),
             color='black', linewidth=0, marker='_', markersize=13)
ax1.get_legend().set_visible(False)
# then, shuffled distribution next to it
sns.violinplot(x=np.concatenate((np.zeros(nshuf), np.ones(nshuf))),
              y=np.concatenate((np.empty((nshuf))*np.nan, full_biasshift_perlab_shuffled)),
               palette=[[1,1,1]], ax=ax1)
ax1.set(ylabel='Within-lab ''bias shift'' consistency', xlabel='', ylim=[0, 200])
ax1.set_xticklabels(['Data', 'Shuffle'], ha='center')
plt.tight_layout()
sns.despine(trim=True)
f.savefig(os.path.join(figpath, "across_mouse_biasshift_var_full.pdf"))

# WHAT IS THE P-VALUE COMPARED TO THE  NULL DISTRIBUTION?
pval = np.mean(full_biasshift_perlab_shuffled > full_biasshift_perlab['biasshift_var'].mean())
print('Full task, biasshift consistency: p-value = %.4f'%pval)

