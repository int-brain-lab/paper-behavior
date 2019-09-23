"""
SIGMOIDAL LEARNING CURVES DURING TRAINING
Anne Urai, CSHL, 2019
"""

from fit_learning_curves import *
import pandas as pd
import numpy as np
import sys
import os
import matplotlib.pyplot as plt
import seaborn as sns
from paper_behavior_functions import query_subjects, seaborn_style
import datajoint as dj
from IPython import embed as shell  # for debugging

# import wrappers etc
from ibl_pipeline import reference, subject, action, acquisition, data, behavior
from ibl_pipeline.utils import psychofit as psy
from ibl_pipeline.analyses import behavior as behavioral_analyses

sys.path.insert(0, '../python')

# INITIALIZE A FEW THINGS
seaborn_style()
figpath = os.path.join(os.path.expanduser('~'), 'Data', 'Figures_IBL')
cmap = sns.diverging_palette(20, 220, n=3, center="dark")
sns.set_palette("gist_gray")  # palette for water types

# ================================= #
# GET DATA FROM TRAINED ANIMALS
# ================================= #

use_subjects = query_subjects()

b = (use_subjects * (acquisition.Session.proj(session_date='DATE(session_start_time)'))
     * behavior.TrialSet * acquisition.Session
     * behavioral_analyses.BehavioralSummaryByDate)  # take all behavior on a given day

behav = pd.DataFrame(b.fetch(order_by='subject_nickname, session_start_time'))
# behav = dj2pandas(bdat)

# which day of training is the mouse in?
behav['session_day'] = behav.groupby(['lab_name', 'subject_nickname'])['session_date'].apply(
    lambda x: (x - np.min(x))).dt.days
behav['session_day'] = behav.groupby(['lab_name', 'subject_nickname'])[
    'session_date'].cumcount()

# # select only those mice for which trainingChoiceWorld is their first session
# # (remove animals that trained on the matlab task beforehand)
# first_protocol = behav[behav.session_day == 0].reset_index()
# remove_subjects = first_protocol.subject_nickname[first_protocol['task_protocol'].isnull()]
# behav = behav[~behav['subject_nickname'].isin(remove_subjects.to_list())]

lab_names = {'cortexlab': 'UCL', 'mainenlab': 'CCU', 'churchlandlab': 'CSHL',
             'wittenlab': 'Princeton', 'angelakilab': 'NYU', 'mrsicflogellab': 'SWC',
             'danlab': 'Berkeley'}

# ================================= #
# LEARNING CURVES
# ================================= #

fig = sns.FacetGrid(behav,
                    col="lab_name", col_wrap=4, col_order=list(lab_names.keys()),
                    sharex=True, sharey=True, aspect=1, xlim=[-1, 50.5], ylim=[0.4, 1])
fig.map(plot_learningcurve, "session_day",
        "performance_easy", "subject_nickname")
fig.set_axis_labels('Days in training', 'Performance on easy trials (%)')
for ax, title in zip(fig.axes.flat, list(lab_names.values())):
    ax.set_title(title)
fig.despine(trim=True)
fig.savefig(os.path.join(figpath, "figure4e_learningcurves_perlab.pdf"))
fig.savefig(os.path.join(
    figpath, "figure4e_learningcurves_perlab.png"), dpi=600)
plt.close('all')

fig = sns.FacetGrid(behav,
                    col="subject_nickname", col_wrap=8, hue="lab_name", palette="colorblind",
                    sharex=True, sharey=True, aspect=1)
fig.map(plot_learningcurve, "session_day",
        "performance_easy", "subject_nickname").add_legend()
fig.set_axis_labels('Days in training', 'Performance on easy trials (%)')
fig.set_titles("{col_name}")
fig.despine(trim=True)
fig.savefig(os.path.join(figpath, "figure4e_learningcurves_permouse.pdf"))
plt.close('all')
