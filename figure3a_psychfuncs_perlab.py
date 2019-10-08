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
seaborn_style()
figpath = figpath()
pal = group_colors()
institution_map, col_names = institution_map()

# ================================= #
# GET DATA FROM TRAINED ANIMALS
# ================================= #

# TODO: WAIT FOR SHAN TO ADD training_day  AND COMPLETE THE QUERY FOR THE RIGHT SESSIONS
use_sessions = query_sessions_around_criterium(days_from_trained=[3, 0])
# restrict by list of dicts with uuids for these sessions
b = acquisition.Session * subject.Subject * subject.SubjectLab * reference.Lab * \
    behavior.TrialSet.Trial & use_sessions[[
        'session_uuid']].to_dict(orient='records')
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
# ================================= #

# plot one curve for each animal, one panel per lab
fig = sns.FacetGrid(behav,
                    col="institution_code", col_wrap=4, col_order=col_names,
                    sharex=True, sharey=True, aspect=1, hue="subject_uuid")
fig.map(plot_psychometric, "signed_contrast", "choice_right",
        "subject_nickname", color='gray', alpha=0.7)
fig.set_titles("{col_name}")
for axidx, ax in enumerate(fig.axes.flat):
    ax.set_title(behav.institution_code.unique()[
                 axidx-1], color=pal[axidx-1], fontweight='bold')

# NOW ADD THE GROUP
ax_group = fig.axes[0]  # overwrite this empty plot
for i, inst in enumerate(behav.institution_code.unique()):
    tmp_behav = behav[behav['institution_code'].str.contains(inst)]
    plot_psychometric(tmp_behav.signed_contrast, tmp_behav.choice_right, tmp_behav.subject_nickname, ax=ax_group, legend=False, color=pal[i])
ax_group.set_title('All labs', color='k', fontweight='bold')
fig.despine(trim=True)
fig.set_axis_labels('Signed contrast (%)', 'Rightward choice (%)')

fig.savefig(os.path.join(figpath, "figure3a_psychfuncs.pdf"))
fig.savefig(os.path.join(figpath, "figure3a_psychfuncs.png"), dpi=600)
plt.close('all')
