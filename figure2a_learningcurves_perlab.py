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

# INITIALIZE A FEW THINGS
seaborn_style()
figpath = figpath()
pal = group_colors()
institution_map, col_names = institution_map()

# ================================= #
# GET DATA FROM TRAINED ANIMALS
# ================================= #

use_subjects = query_sessions()
b = (behavioral_analyses.BehavioralSummaryByDate * use_subjects)
behav = b.fetch(order_by='institution_short, subject_nickname, training_day',
                format='frame').reset_index()
behav['institution_code'] = behav.institution_short.map(institution_map)

# how many mice are there for each lab?
N = behav.groupby(['institution_code'])['subject_nickname'].nunique().to_dict()
behav['n_mice'] = behav.institution_code.map(N)
behav['institution_name'] = behav.institution_code + ': ' + behav.n_mice.apply(str) + ' mice'

# make sure each mouse starts at 0
for index, group in behav.groupby(['lab_name', 'subject_nickname']):
    behav['training_day'][behav.index.isin(
        group.index)] = group['training_day'] - group['training_day'].min()

# ================================= #
# LEARNING CURVES
# ================================= #

# plot one curve for each animal, one panel per lab
fig = sns.FacetGrid(behav,
                    col="institution_code", col_wrap=4, col_order=col_names,
                    sharex=True, sharey=True, aspect=1, hue="subject_uuid", xlim=[-1, 41.5])
fig.map(sns.lineplot, "training_day",
        "performance_easy", color='gray', alpha=0.3)
fig.set_titles("{col_name}")
for axidx, ax in enumerate(fig.axes.flat[0:-1]):
    ax.set_title(behav.institution_name.unique()[axidx], color=pal[axidx], fontweight='bold')

# overlay the example mouse
sns.lineplot(ax=fig.axes[-2], x='training_day', y='performance_easy', color='black', 
             data=behav[behav['subject_nickname'].str.contains('KS014')], legend=False)

# NOW ADD THE GROUP
ax_group = fig.axes[-1] # overwrite this empty plot
sns.lineplot(x='training_day', y='performance_easy', hue='institution_code', palette=pal, 
    ax=ax_group, legend=False, data=behav)
ax_group.set_title('All labs', color='k', fontweight='bold')
fig.set_axis_labels('Training day', 'Performance (%) on easy trials')

fig.despine(trim=True)
fig.savefig(os.path.join(figpath, "figure2a_learningcurves.pdf"))
fig.savefig(os.path.join(figpath, "figure2a_learningcurves.png"), dpi=600)
