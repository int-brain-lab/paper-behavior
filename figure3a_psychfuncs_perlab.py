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
col_names = col_names[:-1]

# ================================= #
# GET DATA FROM TRAINED ANIMALS
# ================================= #

use_sessions, use_days = query_sessions_around_criterion(criterion='trained',
                                                         days_from_criterion=[2, 0],
                                                         as_dataframe=False)
# restrict by list of dicts with uuids for these sessions
b = use_sessions * subject.Subject * subject.SubjectLab * reference.Lab * \
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
# ================================= #

# how many mice are there for each lab?
N = behav.groupby(['institution_code'])['subject_nickname'].nunique().to_dict()
behav['n_mice'] = behav.institution_code.map(N)
behav['institution_name'] = behav.institution_code + ': ' + behav.n_mice.apply(str) + ' mice'

# plot one curve for each animal, one panel per lab
fig = sns.FacetGrid(behav,
                    col="institution_code", col_wrap=7, col_order=col_names,
                    sharex=True, sharey=True, aspect=0.7, hue="subject_uuid")
fig.map(plot_psychometric, "signed_contrast", "choice_right",
        "subject_nickname", color='gray', alpha=0.7)
fig.set_titles("{col_name}")
for axidx, ax in enumerate(fig.axes.flat):
    ax.set_title(sorted(behav.institution_name.unique())[axidx], color=pal[axidx], fontweight='bold')

# overlay the example mouse
tmpdat = behav[behav['subject_nickname'].str.contains('KS014')]
plot_psychometric(tmpdat.signed_contrast, tmpdat.choice_right, tmpdat.subject_nickname,
                  color='black', ax=fig.axes[0], legend=False)
fig.despine(trim=True)
fig.set_axis_labels('Signed contrast (%)', 'Rightward choice (%)')

fig.savefig(os.path.join(figpath, "figure3a_psychfuncs.pdf"))
fig.savefig(os.path.join(figpath, "figure3a_psychfuncs.png"), dpi=300)

# Plot all labs
fig, ax1 = plt.subplots(1, 1, figsize=(4, 4))
for i, inst in enumerate(behav.institution_code.unique()):
    tmp_behav = behav[behav['institution_code'].str.contains(inst)]
    plot_psychometric(tmp_behav.signed_contrast, tmp_behav.choice_right,
                      tmp_behav.subject_nickname, ax=ax1, legend=False, color=pal[i])
ax1.set_title('All labs', color='k', fontweight='bold')
ax1.set(xlabel='Signed contrast (%)', ylabel='Rightward choice (%)')
seaborn_style()
plt.tight_layout(pad=2)

fig.savefig(os.path.join(figpath, "figure3a_psychfuncs_all_labs.pdf"))
fig.savefig(os.path.join(figpath, "figure3a_psychfuncs_all_labs.png"), dpi=300)
