"""
PSYCHOMETRIC AND CHRONOMETRIC FUNCTIONS OF TRAINED ANIMALS
Anne Urai, CSHL, 2019
"""

import pandas as pd
import numpy as np
import seaborn as sns
import sys
import os
import matplotlib.pyplot as plt
from paper_behavior_functions import *
import datajoint as dj
from IPython import embed as shell  # for debugging
from scipy.special import erf  # for psychometric functions

# import wrappers etc
from ibl_pipeline import reference, subject, action, acquisition, data, behavior
from ibl_pipeline.utils import psychofit as psy
from ibl_pipeline.analyses import behavior as behavioral_analyses
from dj_tools import *

# INITIALIZE A FEW THINGS
sns.set(style="ticks", context="paper", font_scale=1.2)
figpath = os.path.join(os.path.expanduser('~'), 'Data', 'Figures_IBL')
cmap = sns.diverging_palette(20, 220, n=3, center="dark")
sns.set_palette("hustl")
pal = sns.color_palette("colorblind", 7)

# ================================= #
# GET DATA FROM TRAINED ANIMALS
# ================================= #

# TODO: WAIT FOR SHAN TO ADD training_day  AND COMPLETE THE QUERY FOR THE RIGHT SESSIONS
use_sessions = query_sessions_around_criterium(days_from_trained=[3, 0])
# restrict by list of dicts with uuids for these sessions
b = acquisition.Session * subject.Subject * subject.SubjectLab * reference.Lab * \
    behavior.TrialSet.Trial & use_sessions[['session_uuid']].to_dict(orient='records')           
bdat = b.fetch(order_by='institution_short, subject_nickname, session_start_time, trial_id', format='frame').reset_index()
behav = dj2pandas(bdat)
assert(~behav.empty)

# ================================= #
# PSYCHOMETRIC FUNCTIONS
# ================================= #

fig = sns.FacetGrid(behav,
	hue="institution_short", palette=pal,
	sharex=True, sharey=True, aspect=1)
fig.map(plot_psychometric, "signed_contrast", "choice_right", "subject_nickname")
fig.set_axis_labels('Signed contrast (%)', 'Rightward choice (%)')
# fig.set_titles("{col_name}")
fig.despine(trim=True)
fig.savefig(os.path.join(figpath, "figure5a_psychfuncs_perlab.pdf"))
fig.savefig(os.path.join(figpath, "figure5a_psychfuncs_perlab.png"), dpi=600)
plt.close('all')

fig = sns.FacetGrid(behav,
	col="institution_short", col_wrap=4, 
	sharex=True, sharey=True, aspect=1, hue="subject_nickname", palette='gist_grey')
fig.map(plot_psychometric, "signed_contrast", "choice_right", "subject_nickname")
fig.set_axis_labels('Signed contrast (%)', 'Rightward choice (%)')
fig.set_titles("{col_name}")
for ax, color in zip(fig.axes.flat, pal):
    ax.set_title(color=color)
fig.despine(trim=True)
fig.savefig(os.path.join(figpath, "figure5a_psychfuncs_perlab_singlemouse.pdf"))
fig.savefig(os.path.join(figpath, "figure5a_psychfuncs_perlab_singlemouse.png"), dpi=600)
plt.close('all')

shell()

# ================================= #
# CHRONOMETRIC FUNCTIONS
# ================================= #
fig = sns.FacetGrid(behav,
	hue="institution_short", palette=pal,
	sharex=True, sharey=True, aspect=1)
fig.map(plot_chronometric, "signed_contrast", "rt", "subject_nickname")
fig.set_axis_labels('Signed contrast (%)', 'Response time (s)')
fig.despine(trim=True)
fig.savefig(os.path.join(figpath, "figure5b_chronfuncs_perlab.pdf"))
fig.savefig(os.path.join(figpath, "figure5b_chronfuncs_perlab.png"), dpi=600)
plt.close('all')

fig = sns.FacetGrid(behav,
	col="institution_short", col_wrap=4, 
	sharex=True, sharey=True, aspect=1, hue="subject_nickname", palette='gist_grey')
fig.map(plot_chronometric, "signed_contrast", "rt", "subject_nickname")
fig.set_axis_labels('Signed contrast (%)', 'Response time (s)')
for ax, color in zip(fig.axes.flat, pal):
    ax.set_title(color=color)
fig.despine(trim=True)
fig.savefig(os.path.join(figpath, "figure5b_chronfuncs_permouse.pdf"))
plt.close('all')