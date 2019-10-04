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
pal = sns.color_palette("colorblind", 7)
institution_map = institution_map()

# ================================= #
# GET DATA FROM TRAINED ANIMALS
# ================================= #

# TODO: WAIT FOR SHAN TO ADD training_day  AND COMPLETE THE QUERY FOR THE RIGHT SESSIONS
use_sessions = query_sessions_around_ephys(days_from_trained=[3, 0])
# restrict by list of dicts with uuids for these sessions
b = acquisition.Session * subject.Subject * subject.SubjectLab * reference.Lab * \
    behavior.TrialSet.Trial & use_sessions[[
        'session_uuid']].to_dict(orient='records')
bdat = b.fetch(order_by='institution_short, subject_nickname, session_start_time, trial_id',
               format='frame').reset_index()
behav = dj2pandas(bdat)
behav['institution_code'] = behav.institution_short.map(institution_map)

assert(~behav.empty)
print(behav.describe())

# # add a fake subject_num to make the colormap restart within each lab
# behav['subject_num'] = behav['subject_uuid'].astype("category").cat.codes
# for index, group in behav.groupby(['institution_short']):
#     behav['subject_num'][behav.index.isin(
#         group.index)] = group['subject_uuid'].astype("category").cat.codes

# ================================= #
# PSYCHOMETRIC FUNCTIONS
# one for all labs combined
# ================================= #

fig = sns.FacetGrid(behav,
                    hue="probabilityLeft", palette=cmap,
                    sharex=True, sharey=True, aspect=1)
fig.map(plot_psychometric, "signed_contrast",
        "choice_right", "subject_nickname")
fig.set_axis_labels('Signed contrast (%)', 'Rightward choice (%)')
fig.ax.annotate('80/20', xy=(-5, 0.6), xytext=(-15, 0.8), color=cmap[0], fontsize=12,
                arrowprops=dict(facecolor=cmap[0], shrink=0.05), ha='right')
fig.ax.annotate('20/80', xy=(5, 0.4), xytext=(15, 0.2), color=cmap[2], fontsize=12,
                arrowprops=dict(facecolor=cmap[2], shrink=0.05))
fig.ax.annotate('20/80', xy=(5, 0.4), xytext=(15, 0.2), color=cmap[2], fontsize=12,
                arrowprops=dict(facecolor=cmap[2], shrink=0.05))
fig.despine(trim=True)
fig.savefig(os.path.join(figpath, "figure4b_psychfuncs_biased_alllabs.pdf"))
fig.savefig(os.path.join(
    figpath, "figure4b_psychfuncs_biased_alllabs.png"), dpi=600)
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
    group2 = group.loc[group.index.repeat(len(yvec))].reset_index(drop=True).copy()
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
fig = sns.FacetGrid(behav3,
                    hue="institution_code", palette=pal,
                    sharex=True, sharey=True, aspect=1)
fig.map(plot_chronometric, "signed_contrast", "biasshift", "subject_nickname")
fig.set_axis_labels('Signed contrast (%)', 'Bias shift ($\Delta$ choice %)')
fig.despine(trim=True)
fig.savefig(os.path.join(figpath, "figure4b_psychfuncs_biasshift_alllabs.pdf"))
fig.savefig(os.path.join(figpath, "figure4b_psychfuncs_biasshift_alllabs.png"), dpi=600)
plt.close('all')

shell()

# plot one curve for each animal, one panel per lab
fig = sns.FacetGrid(behav3,
	col="institution_code", col_wrap=4, 
	sharex=True, sharey=True, aspect=1, hue="subject_uuid")
fig.map(plot_chronometric, "signed_contrast", "biasshift", "subject_nickname", color='gray', alpha=0.7)
fig.set_axis_labels('Signed contrast (%)', 'Bias shift ($\Delta$ choice %)')
fig.set_titles("{col_name}")
for axidx, ax in enumerate(fig.axes.flat):
    ax.set_title(behav.institution_code.unique()[axidx], color=pal[axidx], fontweight='bold')
fig.despine(trim=True)
fig.savefig(os.path.join(figpath, "figure3a_psychfuncs_perlab_singlemouse.pdf"))
fig.savefig(os.path.join(figpath, "figure3a_psychfuncs_perlab_singlemouse.png"), dpi=600)
plt.close('all')