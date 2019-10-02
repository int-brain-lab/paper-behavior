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

sys.path.insert(0, '../python')

# INITIALIZE A FEW THINGS
sns.set(style="ticks", context="paper", font_scale=1.2)
figpath = figpath()
cmap = sns.diverging_palette(20, 220, n=3, center="dark")
sns.set_palette(cmap)  # palette for water types
pal = sns.color_palette("colorblind", 7)

# ================================= #
# GET DATA FROM TRAINED ANIMALS
# ================================= #

# TODO: WAIT FOR SHAN TO ADD training_day  AND COMPLETE THE QUERY FOR THE RIGHT SESSIONS
use_sessions = query_sessions(as_dataframe=True)

# restrict by list of dicts with uuids for these sessions
b = (acquisition.Session & 'task_protocol LIKE "%biased%"') \
     * subject.Subject * subject.SubjectLab * reference.Lab * \
     behavior.TrialSet.Trial & use_sessions[['session_uuid']].to_dict(orient='records')
bdat = b.fetch(order_by='institution_short, subject_nickname, session_start_time, trial_id', format='frame').reset_index()
behav = dj2pandas(bdat)
assert(~behav.empty)
print(behav.describe())

# add a fake subject_num to make the colormap restart within each lab
behav['subject_num'] = behav['subject_uuid'].astype("category").cat.codes
for index, group in behav.groupby(['institution_short']):
      behav['subject_num'][behav.index.isin(group.index)] = group['subject_uuid'].astype("category").cat.codes

# ================================= #
# PSYCHOMETRIC FUNCTIONS
# one for all labs combined
# ================================= #

fig = sns.FacetGrid(behav,
	hue="probabilityLeft", palette=cmap,
	sharex=True, sharey=True, aspect=1)
fig.map(plot_psychometric, "signed_contrast", "choice_right", "subject_nickname")
fig.set_axis_labels('Signed contrast (%)', 'Rightward choice (%)')
ax.annotate('80/20', xy=(-20, 50), xytext=(-40, 80), color=cmap[0], fontsize=12,
    arrowprops=dict(facecolor=cmap[0], shrink=0.05), ha='right')
ax.annotate('20/80', xy=(20, 50), xytext=(40, 20), color=cmap[2], fontsize=12,
    arrowprops=dict(facecolor=cmap[2], shrink=0.05))

fig.despine(trim=True)
fig.savefig(os.path.join(figpath, "figure4b_psychfuncs_biased_alllabs.pdf"))
fig.savefig(os.path.join(figpath, "figure4b_psychfuncs_biased_alllabs.png"), dpi=600)
plt.close('all')

shell()

# ================================================================== #
# DIFFERENCE BETWEEN TWO PSYCHOMETRIC FUNCTIONS
# FOR EACH ANIMAL + for each lab (in 'lab color')
# ================================================================== #

df2 = behav.groupby(['signed_contrast', 'subject_nickname', 'probabilityLeft']).agg(
                {'choice2': 'count', 'choice': 'mean'}).reset_index()
df2.rename(columns={"choice2": "ntrials",
                "choice": "fraction"}, inplace=True)
df2 = df2.groupby(['signed_contrast']).mean().reset_index()
df2 = df2[['signed_contrast', 'ntrials', 'fraction']]

# plot one curve for each animal, one panel per lab
fig = sns.FacetGrid(behav2,
	col="institution_short", col_wrap=4, 
	sharex=True, sharey=True, aspect=1, hue="subject_num", palette="gist_gray")
fig.map(plot_psychometric, "signed_contrast", "choice_right", "subject_nickname")
fig.set_axis_labels('Signed contrast (%)', '$\Delta$ choice (%)')
fig.set_titles("{col_name}")
for axidx, ax in enumerate(fig.axes.flat):
    ax.set_title(behav.institution_short.unique()[axidx], color=pal[axidx])
fig.despine(trim=True)
fig.savefig(os.path.join(figpath, "figure4b_psychfuncs_biased_perlab_singlemouse.pdf"))
fig.savefig(os.path.join(figpath, "figure4b_psychfuncs_biased_perlab_singlemouse.png"), dpi=600)
plt.close('all')
