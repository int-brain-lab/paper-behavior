"""
PSYCHOMETRIC AND CHRONOMETRIC FUNCTIONS IN BIASED BLOCKS
Anne Urai, CSHL, 2019
"""

from dj_tools import *
import pandas as pd
import numpy as np
import sys
import os
import matplotlib.pyplot as plt
import seaborn as sns
from paper_behavior_functions import *
import datajoint as dj
from IPython import embed as shell  # for debugging
from scipy.special import erf  # for psychometric functions

# import wrappers etc
from ibl_pipeline import reference, subject, action, acquisition, data, behavior
from ibl_pipeline.utils import psychofit as psy
from ibl_pipeline.analyses import behavior as behavioral_analyses
from ibl_pipeline.utils import psychofit as psy

sys.path.insert(0, '../python')

# INITIALIZE A FEW THINGS
sns.set(style="ticks", context="paper", font_scale=1.2)
figpath = figpath()
cmap = sns.diverging_palette(20, 220, n=3, center="dark")
sns.set_palette(cmap)  # palette for water types
pal = group_colors()
institution_map, col_names = institution_map()

# ================================= #
# PSYCHOMETRIC FUNCTIONS
# FOR OUR EXAMPLE ANIMAL
# ================================= #

b = (subject.Subject & 'subject_nickname="KS014"') * \
    behavior.TrialSet.Trial * \
    (acquisition.Session & 'task_protocol LIKE "%biased%"')
bdat = b.fetch(order_by='session_start_time, trial_id',
               format='frame').reset_index()
behav = dj2pandas(bdat)
assert not behav.empty

fig = sns.FacetGrid(behav,
                    hue="probabilityLeft", palette=cmap,
                    sharex=True, sharey=True, aspect=1)
fig.map(plot_psychometric, "signed_contrast", "choice_right", "session_uuid")
fig.set_axis_labels('Signed contrast (%)', 'Rightward choice (%)')
fig.ax.annotate('80/20', xy=(-5, 0.6), xytext=(-15, 0.8), color=cmap[0], fontsize=12,
                arrowprops=dict(facecolor=cmap[0], shrink=0.05), ha='right')
fig.ax.annotate('20/80', xy=(5, 0.4), xytext=(13, 0.18), color=cmap[2], fontsize=12,
                arrowprops=dict(facecolor=cmap[2], shrink=0.05))
fig.despine(trim=True)
fig.axes[0][0].set_title('Example mouse', fontweight='bold', color='k')
fig.savefig(os.path.join(figpath, "figure4b_psychfuncs_biased_example.pdf"))
fig.savefig(os.path.join(
    figpath, "figure4b_psychfuncs_biased_example.png"), dpi=600)
plt.close('all')

# ================================= #
# GET DATA FROM TRAINED ANIMALS
# ================================= #

use_sessions, use_days = query_sessions_around_criterion(criterion='ephys',
                                                         days_from_criterion=[
                                                             2, 0],
                                                         as_dataframe=False)
# restrict by list of dicts with uuids for these sessions
b = (use_sessions & 'task_protocol LIKE "%biased%"') \
    * subject.Subject * subject.SubjectLab * reference.Lab * \
    behavior.TrialSet.Trial
# reduce the size of the fetch
b2 = b.proj('institution_short', 'subject_nickname', 'task_protocol',
            'trial_stim_contrast_left', 'trial_stim_contrast_right', 'trial_response_choice',
            'task_protocol', 'trial_stim_prob_left', 'trial_feedback_type')
bdat = b2.fetch(order_by='institution_short, subject_nickname, session_start_time, trial_id',
                format='frame').reset_index()
behav = dj2pandas(bdat)
behav['institution_code'] = behav.institution_short.map(institution_map)

assert(~behav.empty)
print(behav.describe())

# how many mice are there for each lab?
N = behav.groupby(['institution_code'])['subject_nickname'].nunique().to_dict()
behav['n_mice'] = behav.institution_code.map(N)
behav['institution_name'] = behav.institution_code + ': ' + behav.n_mice.apply(str) + ' mice'

# ================================= #
# PSYCHOMETRIC FUNCTIONS
# one for all labs combined
# ================================= #

fig = sns.FacetGrid(behav,
                    hue="probabilityLeft", palette=cmap,
                    sharex=True, sharey=True, aspect=1)
fig.map(plot_psychometric, "signed_contrast", "choice_right", "subject_nickname")
fig.set_axis_labels('Signed contrast (%)', 'Rightward choice (%)')
fig.ax.annotate('80/20', xy=(-5, 0.6), xytext=(-15, 0.8), color=cmap[0], fontsize=12,
                arrowprops=dict(facecolor=cmap[0], shrink=0.05), ha='right')
fig.ax.annotate('20/80', xy=(5, 0.4), xytext=(13, 0.18), color=cmap[2], fontsize=12,
                arrowprops=dict(facecolor=cmap[2], shrink=0.05))
fig.despine(trim=True)
fig.axes[0][0].set_title('All mice', fontweight='bold', color='k')
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

##### PLOT ##########

# plot one curve for each animal, one panel per lab
fig = sns.FacetGrid(behav3,
                    col="institution_code", col_wrap=7, col_order=col_names,
                    sharex=True, sharey=True, aspect=1, hue="subject_nickname")
fig.map(plot_chronometric, "signed_contrast", "biasshift",
        "subject_nickname", color='gray', alpha=0.7)
fig.set_axis_labels('Signed contrast (%)', '$\Delta$ Rightward choice (%)')
fig.set_titles("{col_name}")
for axidx, ax in enumerate(fig.axes.flat[0:-1]):
    ax.set_title(sorted(behav.institution_name.unique())[axidx], color=pal[axidx], fontweight='bold')

# overlay the example mouse
tmpdat = behav3[behav3['subject_nickname'].str.contains('KS014')]
plot_chronometric(tmpdat.signed_contrast, tmpdat.biasshift, tmpdat.subject_nickname,
                  color='black', ax=fig.axes[0], legend=False)

# ADD THE GROUP TO THE FIRST AXIS
ax_group = fig.axes[-1]  # overwrite this empty plot
for i, inst in enumerate(behav.institution_code.unique()):
    tmp_behav = behav3[behav3['institution_code'].str.contains(inst)]
    plot_chronometric(tmp_behav.signed_contrast, tmp_behav.biasshift,
                      tmp_behav.subject_nickname, ax=ax_group, legend=False, color=pal[i])
ax_group.set_title('All labs', color='k', fontweight='bold')
fig.set_axis_labels('Signed contrast (%)', '$\Delta$ Rightward choice (%)')
fig.despine(trim=True)
fig.savefig(os.path.join(figpath, "figure4c_biasshift.pdf"))
fig.savefig(os.path.join(figpath, "figure4c_biasshift.png"), dpi=300)
plt.close('all')

# ================================================================== #
# Plot behavioral metrics per lab
# ================================================================== #

bias = behav3.loc[behav3.signed_contrast == 0, :]
# Add all mice to dataframe seperately for plotting
bias_all = bias.copy()
bias_all['institution_code'] = 'All'
bias_all = bias.append(bias_all)

# Set color palette
use_palette = [[0.6, 0.6, 0.6]] * len(np.unique(bias['institution_code']))
use_palette = use_palette + [[1, 1, 0.2]]
sns.set_palette(use_palette)

# plot
f, ax1 = plt.subplots(1, 1, figsize=(3.5, 3))
sns.set_palette(use_palette)

sns.boxplot(y='biasshift', x='institution_code', data=bias_all, ax=ax1)
ax1.set(ylabel='$\Delta$ Rightward choice (%)\n at 0% contrast', ylim=[0, 51], xlabel='')
[tick.set_color(pal[i]) for i, tick in enumerate(ax1.get_xticklabels()[:-1])]
plt.setp(ax1.xaxis.get_majorticklabels(), rotation=40)
plt.tight_layout(pad=2)
seaborn_style()
plt.savefig(os.path.join(figpath, 'figure4e_bias_per_lab.pdf'))
plt.savefig(os.path.join(figpath, 'figure4e_bias_per_lab.png'), dpi=300)

