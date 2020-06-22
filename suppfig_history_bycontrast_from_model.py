"""
History-dependent choice strategy, depending on previous contrast
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


# Make simulation for all trials
behav = pd.concat([behav,tbehav])
behav = behav[behav['simulation_prob'].notna()]
behav['simulation'] = np.random.binomial(1, p = behav['simulation_prob'])
behav['real_choice']  = behav['choice'].copy()
behav['choice'] = behav['simulation']
behav['previous_signed_contrast'] = behav['signed_contrast'].shift(1).to_numpy()
behav['next_signed_contrast'] = behav['signed_contrast'].shift(-1).to_numpy()
behav['next_signed_contrast'] = behav['signed_contrast'].shift(-1).to_numpy()
behav['next_choice'] =  behav['choice'].shift(-1).to_numpy()
behav['next_choice'] =  behav['previous_choice'].shift(1).to_numpy()
behav['previous_outcome'] = behav['trial_feedback_type'].shift(1).to_numpy()
behav['next_outcome'] = behav['trial_feedback_type'].shift(-1).to_numpy()

#behav['choice'] = behav['real_choice'].copy()

# Delete first and last row
behav = behav.iloc[1:-1,:]

# ======================================== #
# HEATMAP, WITHOUT PSYCHFUNC FITTING
# see Lak et al. eLife 2020, figure 1f
# ======================================== #

update_training = pd.pivot_table(behav[(behav.task == 'traini') & (behav.previous_outcome == 1)].
                                 groupby(['signed_contrast',
                                          'previous_signed_contrast',
                                          'subject_nickname'])[
                                     'choice'].mean().reset_index(),
                                      values='choice',
                                      index=['signed_contrast'],
                                      columns=['previous_signed_contrast'],
                                      aggfunc='mean')
future_training = pd.pivot_table(behav[(behav.task == 'traini') &
                                       (behav.previous_outcome == 1)].groupby(['signed_contrast',
                                          'next_signed_contrast',
                                          'subject_nickname'])[
                                     'choice'].mean().reset_index(),
                          values='choice',
                          index=['signed_contrast'],
                          columns=['next_signed_contrast'],
                          aggfunc='mean')

update_biased = pd.pivot_table(behav[(behav.task == 'biased') & (behav.previous_outcome == 1)].
                               groupby(['signed_contrast',
                                        'previous_signed_contrast',
                                        'subject_nickname'])[
                                     'choice'].mean().reset_index(),
                               values='choice',
                               index=['signed_contrast'],
                               columns=['previous_signed_contrast'],
                               aggfunc='mean')
future_biased = pd.pivot_table(behav[(behav.task == 'biased') & (behav.previous_outcome == 1)].
                               groupby(['signed_contrast',
                                        'next_signed_contrast',
                                        'subject_nickname'])[
                                     'choice'].mean().reset_index(),
                               values='choice',
                               index=['signed_contrast'],
                               columns=['next_signed_contrast'],
                               aggfunc='mean')
# subtract the average psychfunc, so that current stimulus does not dominate!
behav['dummy'] = 1
avg_psychfunc = pd.pivot_table(behav.groupby(['subject_nickname',
                                              'signed_contrast',
                                              'dummy'])['choice'].mean().reset_index(),
                          values='choice', index=['signed_contrast'],
                          columns=['dummy'], aggfunc='mean')

plt.close('all')
fig, ax = plt.subplots(1, 2, figsize=[8, 3.5])
kwargs = {'linewidths':0, 'cmap':"PuOr", 'square': True,
          'cbar_kws':{'label': 'Updating (%)', 'shrink': 0.8,
                      'ticks': [-0.2, 0, 0.2]},
            'vmin':-0.3, 'vmax':0.3}
sns.heatmap(update_training.sub(future_training.values, axis='rows'),
            ax=ax[0], cbar=False, **kwargs)
ax[0].set(xlabel='Previous signed contrast (%)',
          ylabel='Current signed contrast (%)',
          title='Training task')

sns.heatmap(update_biased.sub(future_biased.values),
            ax=ax[1], cbar=True, **kwargs)
ax[1].set(xlabel='Previous signed contrast (%)', ylabel='',
          yticklabels=[], title='Biased task')
fig.tight_layout()
fig.savefig(os.path.join(figpath, "figure5d_history_prevcontrast_heatmap.pdf"))

# ============================================== #
# DEPENDENCE ON PREVIOUS CONTRAST - SUMMARY
# ============================================== #

def pars2choicefract(group):
    group['choicefract'] = psy.erf_psycho_2gammas([group.bias.item(),
                                       group.threshold.item(),
                                       group.lapselow.item(),
                                       group.lapsehigh.item()], 0) * 100
    return group


print('fitting psychometric functions, NOW ALSO BASED ON FUTURE CONTRAST...')
# for name, gr in behav.groupby(['subject_nickname', 'task', 'next_choice', 'next_outcome', 'next_contrast']):
#     print(gr.reset_index()['subject_nickname'].unique())
#     fit_psychfunc(gr.reset_index())

pars = behav.groupby(['subject_nickname', 'task', 'lab_name',
                      'next_choice', 'next_outcome',
                      'next_signed_contrast']).apply(fit_psychfunc).reset_index()
# convert to choice fraction
pars2 = pars.groupby(['subject_nickname', 'task', 'lab_name',
                      'next_choice', 'next_outcome', 'next_signed_contrast']).apply(
    pars2choicefract).reset_index()
# now compute the dependence on previous choice
future_shift = pd.pivot_table(pars2, values='choicefract',
                       index=['task', 'lab_name', 'subject_nickname',
                              'next_outcome', 'next_signed_contrast'],
                       columns='next_choice').reset_index()
future_shift['future_shift'] = future_shift[1.] - future_shift[-1.]
# rename, so that the columns can be matched with history shift
future_shift = future_shift.rename(columns={'next_outcome': 'previous_outcome',
                                            'next_signed_contrast': 'previous_contrast'})

print('fitting psychometric functions, NOW ALSO BASED ON PREVIOUS CONTRAST...')
pars = behav.groupby(['subject_nickname', 'task', 'lab_name',
                      'previous_choice', 'previous_outcome',
                      'previous_contrast']).apply(fit_psychfunc).reset_index()
# convert to choice fraction
pars2 = pars.groupby(['subject_nickname', 'task', 'lab_name',
                      'previous_choice', 'previous_outcome',
                      'previous_contrast']).apply(pars2choicefract).reset_index()
# now compute the dependence on previous choice
history_shift = pd.pivot_table(pars2, values='choicefract',
                       index=['task', 'lab_name', 'subject_nickname',
                              'previous_outcome', 'previous_contrast'],
                       columns='previous_choice').reset_index()
history_shift['history_shift'] = history_shift[1.] - history_shift[-1.]

# ================================= #
# merge and subtract the future shift from each history shift
# ================================= #

pars5 = pd.merge(history_shift, future_shift,
                 on=['subject_nickname', 'previous_outcome',
                     'previous_contrast', 'task', 'lab_name'])
pars5['history_shift_corrected'] = pars5['history_shift'] - pars5['future_shift']
history_shift = pars5.copy()
history_shift.previous_contrast.replace([100], [40], inplace=True)

# ================================= #
# PLOT PREVIOUS CONTRAST-DEPENDENCE
# ================================= #

plt.close('all')
fig, axes = plt.subplots(1, 2, figsize=[6,3], sharex=True, sharey=True)
for task, taskname, ax in zip(['traini', 'biased'], ['Basic task', 'Biased task'], axes):
    # # thin labels, per lab
    # sns.lineplot(data=history_shift[(history_shift.task == task)].groupby(['lab_name',
    #                                                                        'previous_contrast',
    #                                                                        'previous_outcome'
    #                                                                        ]).mean().reset_index(),
    #              x='previous_contrast', y='history_shift_corrected',
    #              hue='previous_outcome', ax=ax, legend=False,
    #              ci=None, marker=None, hue_order=[-1., 1.],
    #              palette=sns.color_palette(["firebrick", "forestgreen"]),
    #              units='lab_name', estimator=None, zorder=0,
    #              linewidth=1, alpha=0.2)
    # thick, across labs
    sns.lineplot(data=history_shift[(history_shift.task == task)],
                 x='previous_contrast', y='history_shift_corrected',
                 hue='previous_outcome', ax=ax, legend=False, estimator=np.mean,
                 err_style='bars', marker='o', hue_order=[-1., 1.],
                 palette=sns.color_palette(["firebrick", "forestgreen"]),
                 zorder=100, ci=95)
    ax.axhline(color='grey', linestyle=':', zorder=-100)
    ax.set(ylabel='Choice updating (%)\n($\Delta$ Rightward choice, corrected)',
              xlabel='Previous contrast (%)',
              xticks=[0, 6, 12, 25, 40],
              xticklabels=['0', '6', '12', '25', '100'],
              title=taskname,
            ylim=[-20, 20])

sns.despine(trim=True)
fig.tight_layout()
fig.savefig(os.path.join(figpath, "figure5c_history_prevcontrast.pdf"))
fig.savefig(os.path.join(figpath, "figure5c_history_prevcontrast.png"), dpi=600)
plt.close("all")