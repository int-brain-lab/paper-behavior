"""
History-dependent choice strategy across tasks
See also https://elifesciences.org/articles/49834

@author: Anne Urai
10 May 2020
"""

# import wrappers etc
from ibl_pipeline import behavior, subject, reference
import matplotlib.pyplot as plt
from dj_tools import dj2pandas, plot_psychometric, fit_psychfunc
from paper_behavior_functions import (seaborn_style, figpath, query_sessions_around_criterion,
                                      group_colors, institution_map)
import statsmodels.api as sm
from statsmodels.formula.api import ols
import pycircstat
import seaborn as sns
import pandas as pd
import numpy as np
import os
from ibl_pipeline.utils import psychofit as psy

# INITIALIZE A FEW THINGS
seaborn_style()
figpath = figpath()
pal = group_colors()
institution_map, col_names = institution_map()

# ================================= #
# GRAB ALL DATA FROM DATAJOINT
# 3 days before and 3 days after starting biasedChoiceWorld
# ================================= #

use_sessions, use_days = query_sessions_around_criterion(criterion='biased',
                                                         days_from_criterion=[
                                                             2, 3],
                                                         as_dataframe=False)
# restrict by list of dicts with uuids for these sessions
b = (use_sessions * subject.Subject * subject.SubjectLab * reference.Lab
     * behavior.TrialSet.Trial)

# reduce the size of the fetch
b2 = b.proj('institution_short', 'subject_nickname', 'task_protocol',
            'trial_stim_contrast_left', 'trial_stim_contrast_right', 'trial_response_choice',
            'task_protocol', 'trial_stim_prob_left', 'trial_feedback_type')
bdat = b2.fetch(order_by='institution_short, subject_nickname, session_start_time, trial_id',
                format='frame').reset_index()
behav = dj2pandas(bdat)

# also code for the future choice (for correction)
behav['next_choice'] = behav.choice.shift(-1)
behav.loc[behav.next_choice == 0, 'next_choice'] = np.nan
behav['next_outcome'] = behav.trial_feedback_type.shift(-1)
behav.loc[behav.next_outcome == 0, 'next_outcome'] = np.nan
behav['next_contrast'] = np.abs(behav.signed_contrast.shift(-1))
behav['next_choice_name'] = behav['next_choice'].map(
    {-1: 'left', 1: 'right'})
behav['next_outcome_name'] = behav['next_outcome'].map(
    {-1: 'pre_error', 1: 'pre_correct'})
behav['institution_code'] = behav.institution_short.map(institution_map)

# easy to use names for groupby
behav['previous_name'] = behav.previous_outcome_name + \
    ', ' + behav.previous_choice_name
behav['next_name'] = behav.next_outcome_name + \
    ', ' + behav.next_choice_name

# split the two types of task protocols (remove the pybpod version number
behav['task'] = behav['task_protocol'].str[14:20]

# remove weird contrast levels
tmpdat = behav.groupby(['signed_contrast'])['choice'].count().reset_index()
removecontrasts = tmpdat.loc[tmpdat['choice'] < 100, 'signed_contrast']
behav = behav[~behav.signed_contrast.isin(removecontrasts)]

# choose: take only those trials where the objective probability is 0.5???
# behav = behav.loc[behav.probabilityLeft == 50, :]

# ================================= #
# PREVIOUS CHOICE - SUMMARY PLOT
# ================================= #

# plot one curve for each animal, one panel per lab
fig = sns.FacetGrid(behav,
                    col='task', hue='previous_name',
                    sharex=True, sharey=True, aspect=0.7, palette='Paired',
                    hue_order=['post_error, right', 'post_correct, right',
                               'post_error, left', 'post_correct, left'])
fig.map(plot_psychometric, "signed_contrast",
        "choice_right", "subject_nickname")
tasks = ['Unbiased task\n(level 1)', 'Biased task\n(level 2)']
for axidx, ax in enumerate(fig.axes.flat):
    ax.set_title(tasks[axidx], color='k', fontweight='bold')
# fig._legend.set_title('Previous choice')
fig.set_axis_labels('Signed contrast (%)', 'Rightward choice (%)')
fig.despine(trim=True)
fig.savefig(os.path.join(figpath, "figure5a_history_psychfuncs.pdf"))
fig.savefig(os.path.join(figpath, "figure5a_history_psychfuncs.png"), dpi=600)
plt.close('all')

fig = sns.FacetGrid(behav,
                    col='task', hue='next_name',
                    sharex=True, sharey=True, aspect=0.7, palette='Paired',
                    hue_order=['pre_error, right', 'pre_correct, right',
                               'pre_error, left', 'pre_correct, left'])
fig.map(plot_psychometric, "signed_contrast",
        "choice_right", "subject_nickname")
tasks = ['Unbiased task\n(level 1)', 'Biased task\n(level 2)']
for axidx, ax in enumerate(fig.axes.flat):
    ax.set_title(tasks[axidx], color='k', fontweight='bold')
# fig._legend.set_title('Previous choice')
fig.set_axis_labels('Signed contrast (%)', 'Rightward choice (%)')
fig.despine(trim=True)
fig.savefig(os.path.join(figpath, "figure5a_future_psychfuncs.pdf"))
fig.savefig(os.path.join(figpath, "figure5a_future_psychfuncs.png"), dpi=600)
plt.close('all')

# ================================= #
# DEFINE HISTORY SHIFT FOR LAG 1
# ================================= #

# instead of the bias in % contrast, take the choice shift at x = 0
# now read these out at the presented levels of signed contrast
def pars2shift(pars, choicevar, outcomevar):
    pars2 = pd.DataFrame([])
    xvec = behav.signed_contrast.unique()
    for index, group in pars.groupby(['institution_code', 'subject_nickname', 'task',
                                      choicevar, outcomevar]):
        # expand
        yvec = psy.erf_psycho_2gammas([group.bias.item(),
                                       group.threshold.item(),
                                       group.lapselow.item(),
                                       group.lapsehigh.item()], xvec)
        group2 = group.loc[group.index.repeat(
            len(yvec))].reset_index(drop=True).copy()
        group2['signed_contrast'] = xvec
        group2['choice'] = 100 * yvec
        # add this
        pars2 = pars2.append(group2)

    # only pick psychometric functions that were fit on a reasonable number of trials...
    pars2 = pars2[(pars2.ntrials > 50) & (pars2.signed_contrast == 0)]

    # compute history-dependent bias shift
    pars3 = pd.pivot_table(pars2, values='choice',
                           index=['institution_code', 'subject_nickname',
                                  'task', outcomevar],
                           columns=[choicevar]).reset_index()
    pars3['bias_shift'] = pars3.right - pars3.left
    pars4 = pd.pivot_table(pars3, values='bias_shift',
                           index=['institution_code', 'subject_nickname', 'task'],
                           columns=[outcomevar]).reset_index()
    print(pars4.describe())
    return pars4

## COMPUTE FOR EACH MOUSE
print('fitting history-dependent psychometric functions...')
pars = behav.groupby(['institution_code', 'subject_nickname', 'task',
                      'previous_choice_name', 'previous_outcome_name']).apply(
    fit_psychfunc).reset_index()
history_shift = pars2shift(pars, 'previous_choice_name', 'previous_outcome_name')

print('fitting future-dependent psychometric functions...')
pars = behav.groupby(['institution_code', 'subject_nickname', 'task',
                      'next_choice_name', 'next_outcome_name']).apply(
    fit_psychfunc).reset_index()
future_shift = pars2shift(pars, 'next_choice_name', 'next_outcome_name')

history_shift = history_shift.merge(future_shift, on=['subject_nickname', 'institution_code', 'task'])
history_shift['post_correct_corr'] = history_shift['post_correct'] - history_shift['pre_correct']
history_shift['post_error_corr'] = history_shift['post_error'] - history_shift['pre_error']

# ================================= #
# STRATEGY SPACE
# ================================= #

plt.close('all')
fig, ax = plt.subplots(1, 2, figsize=[8, 3.5])
sns.lineplot(x='post_correct', y='post_error',
             units='subject_nickname', estimator=None, hue='institution_code', alpha=0.3,
             palette=pal, data=history_shift, ax=ax[0], legend=False)
# markers; only for those subjects with all 4 conditions
num_dp = history_shift.groupby(['subject_nickname'])['post_error'].count().reset_index()
sjs = num_dp.loc[num_dp.post_error == 2, 'subject_nickname'].to_list()
sns.lineplot(x='post_correct', y='post_error',
             units='subject_nickname', estimator=None, hue='institution_code', palette=pal,
             alpha=0.5, legend=False, data=history_shift[history_shift['subject_nickname'].isin(sjs)],
             ax=ax[0], style='task', markers={'traini': 'o', 'biased': '^'}, markersize=4)
# add black line for the group
sns.lineplot(x='post_correct', y='post_error', legend=False, color='k', ci=None,
             data=history_shift[history_shift['subject_nickname'].isin(sjs)].groupby(
                 ['task']).mean().reset_index(), ax=ax[0])
sns.lineplot(x='post_correct', y='post_error', legend=False, color='k', ci=None,
             data=history_shift[history_shift['subject_nickname'].isin(sjs)].groupby(
                 ['task']).mean().reset_index(),
             ax=ax[0], style='task', markers={'traini': 'o', 'biased': '^'}, markersize=6)
ax[0].set_xlabel("\u0394 Rightward choices \nafter correct (%)")
ax[0].set_ylabel("\u0394 Rightward choices \nafter error (%)")

ax[0].set_title('Uncorrected')
ax[0].set(xticks=[-20, 0, 20, 40, 60], yticks=[-20, 0, 20, 40, 60])
ax[0].axhline(linestyle=':', color='darkgrey')
ax[0].axvline(linestyle=':', color='darkgrey')


#### REPEATE BUT WITH CORRECTION
sns.lineplot(x='post_correct_corr', y='post_error_corr',
             units='subject_nickname', estimator=None, hue='institution_code', alpha=0.3,
             palette=pal, data=history_shift, ax=ax[1], legend=False)
sns.lineplot(x='post_correct_corr', y='post_error_corr',
             units='subject_nickname', estimator=None, hue='institution_code', palette=pal,
             alpha=0.5, legend=False, data=history_shift[history_shift['subject_nickname'].isin(sjs)],
             ax=ax[1], style='task', markers={'traini': 'o', 'biased': '^'}, markersize=4)
# add black line for the group
sns.lineplot(x='post_correct_corr', y='post_error_corr', legend=False, color='k', ci=None,
             data=history_shift[history_shift['subject_nickname'].isin(sjs)].groupby(
                 ['task']).mean().reset_index(), ax=ax[1])
sns.lineplot(x='post_correct_corr', y='post_error_corr', legend=False, color='k', ci=None,
             data=history_shift[history_shift['subject_nickname'].isin(sjs)].groupby(
                 ['task']).mean().reset_index(),
             ax=ax[1], style='task', markers={'traini': 'o', 'biased': '^'}, markersize=6)
ax[1].set_xlabel("\u0394 Rightward choices \nafter correct (%)")
ax[1].set_ylabel("\u0394 Rightward choices \nafter error (%)")

ax[1].set_title('Corrected')
ax[1].set(xticks=[-20, 0, 20, 40, 60], yticks=[-20, 0, 20, 40, 60])
ax[1].axhline(linestyle=':', color='darkgrey')
ax[1].axvline(linestyle=':', color='darkgrey')

sns.despine(trim=True)
fig.tight_layout()
fig.savefig(os.path.join(figpath, "figure5b_history_strategy.pdf"))
fig.savefig(os.path.join(figpath, "figure5b_history_strategy.png"), dpi=600)
plt.close("all")

# ================================= #
# do stats on this
# ================================= #

# compute the shift by subtracting between the two tasks
pars5 = pd.pivot_table(history_shift, values=['post_correct', 'post_error'],
                       index=['institution_code', 'subject_nickname'],
                       columns=['task']).reset_index()
pars5['coord_shift_x'] = pars5['post_correct']['biased'] - \
    pars5['post_correct']['traini']
pars5['coord_shift_y'] = pars5['post_error']['biased'] - \
    pars5['post_error']['traini']


# convert coordinates to norm and angle
def cart2pol(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return(rho, phi)


r, phi = cart2pol(pars5['coord_shift_x'], pars5['coord_shift_y'])
pars5['norm'] = r
pars5['angle'] = phi
pars5 = pars5.dropna()

# stats on vector norm between laboratories:
sm_lm = ols('norm ~ C(institution_code)', data=pars5).fit()
table = sm.stats.anova_lm(sm_lm)  # Type 2 ANOVA DataFrame
print(table)

# use pycircstat Watson-Williams test
pars6 = pars5.groupby('institution_code')['angle'].aggregate(
    lambda x: list(x)).reset_index()
angles_grouped = pars6['angle'].values

pval, table = pycircstat.watson_williams(angles_grouped[0],
                                         angles_grouped[1],
                                         angles_grouped[2],
                                         angles_grouped[3],
                                         angles_grouped[4],
                                         angles_grouped[5],
                                         angles_grouped[6])
print('circular one-way anova')
print(table)

# fig, ax = plt.subplots(2, 1)
# sns.swarmplot(x='institution_code', y='norm', data=pars5, ax=ax[0])
# sns.swarmplot(x='institution_code', y='angle', data=pars5, ax=ax[1])
# fig.tight_layout()
# fig.savefig(os.path.join(figpath, "history_shift_stats.pdf"))
# fig.savefig(os.path.join(figpath, "history_shift_stats.png"), dpi=600)
# plt.close("all")
