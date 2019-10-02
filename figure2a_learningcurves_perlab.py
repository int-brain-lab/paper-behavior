"""
SIGMOIDAL LEARNING CURVES DURING TRAINING
Anne Urai, CSHL, 2019
"""

import pandas as pd
import numpy as np
import sys
import os
import matplotlib.pyplot as plt
import seaborn as sns
from paper_behavior_functions import *
import datajoint as dj
from IPython import embed as shell  # for debugging

# import wrappers etc
from ibl_pipeline import reference, subject, action, acquisition, data, behavior
from ibl_pipeline.utils import psychofit as psy
from ibl_pipeline.analyses import behavior as behavioral_analyses

sys.path.insert(0, '../analysis_IBL/python')
from fit_learning_curves import *

# INITIALIZE A FEW THINGS
seaborn_style()
figpath = figpath()
pal = sns.color_palette("colorblind", 7)

# ================================= #
# GET DATA FROM TRAINED ANIMALS
# ================================= #

use_subjects = query_sessions()
b = (behavioral_analyses.BehavioralSummaryByDate * use_subjects)
behav = b.fetch(order_by='institution_short, subject_nickname, training_day', format='frame').reset_index()

# ================================= #
# LEARNING CURVES
# ================================= #

fig = sns.FacetGrid(behav,
	hue="institution_short", palette=pal,
	sharex=True, sharey=True, aspect=1)
fig.map(sns.lineplot, "training_day", "performance_easy")
fig.set_axis_labels('Days in training', 'Performance on easy trials (%)')
fig.despine(trim=True)
fig.savefig(os.path.join(figpath, "figure2a_learningcurves_perlab.pdf"))
fig.savefig(os.path.join(figpath, "figure2a_learningcurves_perlab.png"), dpi=600)
plt.close('all')

fig = sns.FacetGrid(behav,
	col="institution_short", col_wrap=4, 
	sharex=True, sharey=True, aspect=1, hue="subject_nickname", palette='gist_gray')
fig.map(sns.lineplot, "training_day", "performance_easy", lw=1)
# add an indication of when each subject is trained
fig.map(sns.lineplot, "training_day", "performance_easy_trained", lw=2)
fig.set_axis_labels('Days in training', 'Performance on easy trials (%)')
fig.set_titles("{col_name}")
for axidx, ax in enumerate(fig.axes.flat):
    ax.set_title(behav.institution_short.unique()[axidx], color=pal[axidx])
fig.despine(trim=True)
fig.savefig(os.path.join(figpath, "figure2a_learningcurves_perlab_singlemouse.pdf"))
fig.savefig(os.path.join(figpath, "figure2a_learningcurves_perlab_singlemouse.png"), dpi=600)
plt.close('all')