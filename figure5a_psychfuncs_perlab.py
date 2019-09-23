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
from paper_behavior_functions import query_subjects, query_sessions, seaborn_style
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
sns.set_palette("gist_gray")  # palette for water types

# ================================= #
# GET DATA FROM TRAINED ANIMALS
# ================================= #

use_subjects = subject.Subject * subject.SubjectLab * subject.SubjectProject & 'subject_project = "ibl_neuropixel_brainwide_01"'
criterion = behavioral_analyses.SessionTrainingStatus()
sess = ((acquisition.Session & 'task_protocol LIKE "%trainingChoiceWorld%"') \
		* (criterion & 'training_status LIKE "trained%"')) * use_subjects

b = (behavior.TrialSet.Trial & sess) * subject.Subject() * subject.SubjectLab()
bdat = b.fetch(order_by='subject_nickname, session_start_time, trial_id', format='frame').reset_index()
behav = dj2pandas(bdat)
assert(~behav.empty)

behav['lab_name'] = behav['lab_name'].str.replace('zadorlab', 'churchlandlab')
behav['lab_name'] = behav['lab_name'].str.replace('hoferlab', 'mrsicflogellab')

lab_names = {'danlab':'Berkeley', 'mainenlab':'CCU', 'churchlandlab':'CSHL', 'cortexlab':'UCL',
			 'angelakilab':'NYU', 'wittenlab':'Princeton', 'mrsicflogellab':'SWC'}
print('lets go')

# ================================= #
# PSYCHOMETRIC FUNCTIONS
# ================================= #

fig = sns.FacetGrid(behav,
	col="lab_name", col_wrap=4, col_order=list(lab_names.keys()),
	sharex=True, sharey=True, aspect=1)
fig.map(plot_psychometric, "signed_contrast", "choice_right", "subject_nickname")
fig.set_axis_labels('Signed contrast (%)', 'Rightward choice (%)')
for ax, title in zip(fig.axes.flat, list(lab_names.values())):
    ax.set_title(title)
fig.despine(trim=True)
fig.savefig(os.path.join(figpath, "figure4a_psychfuncs_perlab.pdf"))
fig.savefig(os.path.join(figpath, "figure4a_psychfuncs_perlab.png"), dpi=600)
plt.close('all')

fig = sns.FacetGrid(behav,
	col="lab_name", col_wrap=4, col_order=list(lab_names.keys()),
	sharex=True, sharey=True, aspect=1, hue="subject_nickname")
fig.map(plot_psychometric, "signed_contrast", "choice_right", "subject_nickname")
fig.set_axis_labels('Signed contrast (%)', 'Rightward choice (%)')
for ax, title in zip(fig.axes.flat, list(lab_names.values())):
    ax.set_title(title)
fig.despine(trim=True)
fig.savefig(os.path.join(figpath, "figure4a_psychfuncs_perlab_singlemouse.pdf"))
fig.savefig(os.path.join(figpath, "figure4a_psychfuncs_perlab_singlemouse.png"), dpi=600)
plt.close('all')

shell()

fig = sns.FacetGrid(behav,
	col="lab_name", col_wrap=1, col_order=list(lab_names.keys()),
	sharex=True, sharey=True, height=2, aspect=1.9)
fig.map(plot_psychometric, "signed_contrast", "choice_right", "subject_nickname")
fig.set_axis_labels('Signed contrast (%)', 'Rightward choice (%)')
for ax, title in zip(fig.axes.flat, list(lab_names.values())):
    ax.set_title(title)
fig.despine(trim=True)
fig.savefig(os.path.join(figpath, "figure4a_psychfuncs_perlab_vertical.pdf"))
# fig.savefig(os.path.join(figpath, "figure4a_psychfuncs_perlab_vertical.png"), dpi=600)
plt.close('all')

fig = sns.FacetGrid(behav,
	col="subject_nickname", col_wrap=8, hue="lab_name", palette="colorblind",
					sharex=True, sharey=True, aspect=1)
fig.map(plot_psychometric, "signed_contrast", "choice_right", "subject_nickname").add_legend()
fig.set_axis_labels('Signed contrast (%)', 'Rightward choice (%)')
fig.set_titles("{col_name}")
fig.despine(trim=True)
fig.savefig(os.path.join(figpath, "figure4a_psychfuncs_permouse.pdf"))
plt.close('all')

# ================================= #
# CHRONOMETRIC FUNCTIONS
# ================================= #

fig = sns.FacetGrid(behav,
	col="lab_name", col_wrap=4, col_order=list(lab_names.keys()),
	palette="gist_gray", sharex=True, sharey=True, aspect=1)
fig.map(plot_chronometric, "signed_contrast", "rt", "subject_nickname")
fig.set_axis_labels('Signed contrast (%)', 'Response time (s)')
for ax, title in zip(fig.axes.flat, list(lab_names.values())):
    ax.set_title(title)
fig.despine(trim=True)
fig.savefig(os.path.join(figpath, "figure4b_chronfuncs_perlab.pdf"))
fig.savefig(os.path.join(figpath, "figure4b_chronfuncs_perlab.png"), dpi=600)
plt.close('all')

fig = sns.FacetGrid(behav,
	col="subject_nickname", col_wrap=8, hue="lab_name", palette="colorblind",
					sharex=True, sharey=True, aspect=1)
fig.map(plot_chronometric, "signed_contrast", "rt", "subject_nickname").add_legend()
fig.set_axis_labels('Signed contrast (%)', 'Response time (s)')
fig.set_titles("{col_name}")
fig.despine(trim=True)
fig.savefig(os.path.join(figpath, "figure4b_chronfuncs_permouse.pdf"))
plt.close('all')