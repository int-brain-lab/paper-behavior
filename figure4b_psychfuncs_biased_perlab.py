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
fig.ax.annotate('20/80', xy=(5, 0.4), xytext=(15, 0.2), color=cmap[2], fontsize=12,
                arrowprops=dict(facecolor=cmap[2], shrink=0.05))
fig.ax.annotate('20/80', xy=(5, 0.4), xytext=(15, 0.2), color=cmap[2], fontsize=12,
                arrowprops=dict(facecolor=cmap[2], shrink=0.05))
fig.despine(trim=True)
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
    group2['choice'] = yvec

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
                    col="institution_code", col_wrap=4, col_order=col_names,
                    sharex=True, sharey=True, aspect=1, hue="subject_nickname")
fig.map(plot_chronometric, "signed_contrast", "biasshift",
        "subject_nickname", color='gray', alpha=0.7)
fig.set_axis_labels('Signed contrast (%)', 'Bias shift ($\Delta$ choice %)')
fig.set_titles("{col_name}")
for axidx, ax in enumerate(fig.axes.flat):
    ax.set_title(behav.institution_code.unique()[
                 axidx-1], color=pal[axidx-1], fontweight='bold')

# ADD THE GROUP TO THE FIRST AXIS
ax_group = fig.axes[0]  # overwrite this empty plot
for i, inst in enumerate(behav.institution_code.unique()):
    tmp_behav = behav3[behav3['institution_code'].str.contains(inst)]
    plot_chronometric(tmp_behav.signed_contrast, tmp_behav.biasshift,
                      tmp_behav.subject_nickname, ax=ax_group, legend=False, color=pal[i])
ax_group.set_title('All labs', color='k', fontweight='bold')
fig.set_axis_labels('Signed contrast (%)', 'Bias shift ($\Delta$ choice %)')
fig.despine(trim=True)
fig.savefig(os.path.join(figpath, "figure4a_biasshift.pdf"))
fig.savefig(os.path.join(figpath, "figure4a_biasshift.png"), dpi=600)
plt.close('all')
