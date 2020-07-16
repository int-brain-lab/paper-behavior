"""
Psychometric function and choice shifts in the biased task

@author: Anne Urai
15 January 2020
"""

from dj_tools import dj2pandas, plot_psychometric, fit_psychfunc, plot_chronometric
import pandas as pd
import numpy as np
import os
from os.path import join
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.formula.api import ols
from paper_behavior_functions import (seaborn_style, figpath, group_colors, institution_map,
                                      query_sessions_around_criterion, EXAMPLE_MOUSE,
                                      FIGURE_HEIGHT, FIGURE_WIDTH)
# import wrappers etc
from ibl_pipeline import reference, subject, behavior
from ibl_pipeline.utils import psychofit as psy

# whether to query data from DataJoint (True), or load from disk (False)
query = True

# Initialize
seaborn_style()
figpath = figpath()
pal = group_colors()
institution_map, col_names = institution_map()
col_names = col_names[:-1]
cmap = sns.diverging_palette(20, 220, n=3, center="dark")
sns.set_palette(cmap)  # palette for water types

# ================================= #
# GET DATA FROM TRAINED ANIMALS
# ================================= #

if query is True:
    # query sessions
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
else:
    behav = pd.read_csv(join('data', 'Fig4.csv'))

# how many mice are there for each lab?
N = behav.groupby(['institution_code'])['subject_nickname'].nunique().to_dict()
behav['n_mice'] = behav.institution_code.map(N)
behav['institution_name'] = behav.institution_code + '\n' + behav.n_mice.apply(str) + ' mice'

# ================================= #
# PSYCHOMETRIC FUNCTIONS
# FOR OUR EXAMPLE ANIMAL
# ================================= #

fig = sns.FacetGrid(behav[behav['subject_nickname'] == EXAMPLE_MOUSE],
                    hue="probabilityLeft", palette=cmap,
                    sharex=True, sharey=True,
                    height=FIGURE_HEIGHT, aspect=(FIGURE_WIDTH/4)/FIGURE_HEIGHT)
fig.map(plot_psychometric, "signed_contrast", "choice_right", "session_uuid")
fig.set_axis_labels('Contrast (%)', 'Rightward choices (%)')
fig.ax.annotate('20:80', xy=(-5, 0.6), xytext=(-25, 0.8), color=cmap[0], fontsize=7)
fig.ax.annotate('80:20', xy=(5, 0.4), xytext=(13, 0.18), color=cmap[2], fontsize=7)
fig.despine(trim=True)
fig.axes[0][0].set_title('Example mouse', fontweight='bold', color='k')
fig.savefig(os.path.join(figpath, "figure4b_psychfuncs_biased_example.pdf"))
fig.savefig(os.path.join(
    figpath, "figure4b_psychfuncs_biased_example.png"), dpi=600)
plt.close('all')

# ================================= #
# PSYCHOMETRIC FUNCTIONS
# one for all labs combined
# ================================= #

fig = sns.FacetGrid(behav,
                    hue="probabilityLeft", palette=cmap,
                    sharex=True, sharey=True,
                    height=FIGURE_HEIGHT, aspect=(FIGURE_WIDTH/4)/FIGURE_HEIGHT)
fig.map(plot_psychometric, "signed_contrast",
        "choice_right", "subject_nickname")
fig.set_axis_labels('Contrast (%)', '')
fig.ax.annotate('20:80', xy=(-5, 0.6), xytext=(-25, 0.8), color=cmap[0], fontsize=7)
fig.ax.annotate('80:20', xy=(5, 0.4), xytext=(13, 0.18), color=cmap[2], fontsize=7)
fig.despine(trim=True)
fig.axes[0][0].set_title('All mice: n = %d' % behav.subject_nickname.nunique(),
                         fontweight='bold', color='k')
fig.axes[0][0].set(yticklabels=" ")
fig.savefig(os.path.join(figpath, "figure4b_psychfuncs_biased.pdf"))
fig.savefig(os.path.join(
    figpath, "figure4b_psychfuncs_biased.png"), dpi=600)
plt.close('all')

# ================================================================== #
# DIFFERENCE BETWEEN TWO PSYCHOMETRIC FUNCTIONS
# FOR EACH ANIMAL + for each lab (in 'lab color')
# ================================================================== #

print('fitting psychometric functions...')
pars = behav.groupby(['institution_code', 'subject_nickname',
                      'probabilityLeft']).apply(fit_psychfunc).reset_index()
# now read these out at the presented levels of signed contrast
behav2 = pd.DataFrame([])
xvec = behav.signed_contrast.unique()
for index, group in pars.groupby(['institution_code', 'subject_nickname',
                                  'probabilityLeft']):
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
    behav2 = behav2.append(group2)

# now subtract these to compute a bias shift
behav3 = pd.pivot_table(behav2, values='choice',
                        index=['institution_code', 'subject_nickname',
                               'signed_contrast'],
                        columns=['probabilityLeft']).reset_index()
behav3['biasshift'] = behav3[20] - behav3[80]

# %% PLOT

# plot one curve for each animal, one panel per lab
plt.close('all')
fig = sns.FacetGrid(behav3,
                    col="institution_code", col_wrap=7, col_order=col_names,
                    sharex=True, sharey=True, hue="subject_nickname",
                    height=FIGURE_HEIGHT, aspect=(FIGURE_WIDTH/7)/FIGURE_HEIGHT)
fig.map(plot_chronometric, "signed_contrast", "biasshift",
        "subject_nickname", color='gray', alpha=0.7)
fig.set_axis_labels('Contrast (%)', '\u0394 Rightward choices (%)')
fig.set_titles("{col_name}")
for axidx, ax in enumerate(fig.axes.flat):
    ax.set_title(sorted(behav.institution_name.unique())[axidx],
                 color=pal[axidx])

# overlay the example mouse
tmpdat = behav3[behav3['subject_nickname'].str.contains(EXAMPLE_MOUSE)]
plot_chronometric(tmpdat.signed_contrast, tmpdat.biasshift, tmpdat.subject_nickname,
                  color='black', ax=fig.axes[0], legend=False)
fig.set_axis_labels('Contrast (%)', '\u0394 Rightward choices (%)')
fig.despine(trim=True)
plt.tight_layout(w_pad=-1.7)
fig.savefig(os.path.join(figpath, "figure4d_biasshift.pdf"))
fig.savefig(os.path.join(figpath, "figure4d_biasshift.png"), dpi=300)
plt.close('all')


# %% PLOT

fig, ax1 = plt.subplots(1, 1, figsize=(FIGURE_WIDTH/4, FIGURE_HEIGHT))
for i, inst in enumerate(behav.institution_code.unique()):
    tmp_behav = behav3[behav3['institution_code'].str.contains(inst)]
    plot_chronometric(tmp_behav.signed_contrast, tmp_behav.biasshift,
                      tmp_behav.subject_nickname, ax=ax1, legend=False, color=pal[i])
# ax1.set_title('All labs', color='k', fontweight='bold')
ax1.set(xlabel='Contrast (%)', ylabel='\u0394 Rightward choices (%)',
        yticks=[0, 10, 20, 30, 40])
sns.despine(trim=True)
plt.tight_layout()
fig.savefig(os.path.join(figpath, "figure4c_biasshift_all_labs.pdf"))
fig.savefig(os.path.join(figpath, "figure4c_biasshift_all_labs.png"), dpi=300)

# ================================================================== #
# Plot behavioral metrics per lab
# ================================================================== #

bias = behav3.loc[behav3.signed_contrast == 0, :]

# stats on bias shift between laboratories:
sm_lm = ols('biasshift ~ C(institution_code)', data=bias).fit()
table = sm.stats.anova_lm(sm_lm, typ=2)  # Type 2 ANOVA DataFrame
print(table)

# Add all mice to dataframe seperately for plotting
bias_all = bias.copy()

print('average bias shift across all mice: ')
print(bias_all['biasshift'].mean())
bias_all['institution_code'] = 'All'

bias_all = bias.append(bias_all)

# Set color palette
use_palette = [[0.6, 0.6, 0.6]] * len(np.unique(bias['institution_code']))
use_palette = use_palette + [[1, 1, 0.2]]
sns.set_palette(use_palette)

# plot
f, ax1 = plt.subplots(1, 1, figsize=(3, 3.5))
sns.set_palette(use_palette)

sns.boxplot(y='biasshift', x='institution_code', data=bias_all, ax=ax1)
ax1.set(ylabel='\u0394 Rightward choices (%)\n at 0% contrast',
        ylim=[0, 51], xlabel='')
[tick.set_color(pal[i]) for i, tick in enumerate(ax1.get_xticklabels()[:-1])]
plt.setp(ax1.xaxis.get_majorticklabels(), rotation=40)
plt.tight_layout(pad=2)
seaborn_style()
# plt.savefig(os.path.join(figpath, 'figure4e_bias_per_lab.pdf'))
# plt.savefig(os.path.join(figpath, 'figure4e_bias_per_lab.png'), dpi=300)
