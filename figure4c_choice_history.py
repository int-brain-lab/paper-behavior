"""
Plot full psychometric functions as a function of choice history,
and separately for 20/80 and 80/20 blocks
"""

import pandas as pd
import numpy as np
import sys, os, time
import matplotlib.pyplot as plt
import seaborn as sns
from paper_behavior_functions import query_subjects
import datajoint as dj
from IPython import embed as shell # for debugging
from math import ceil
# from figure_style import seaborn_style

# import wrappers etc
from ibl_pipeline import reference, subject, action, acquisition, data, behavior
from ibl_pipeline.utils import psychofit as psy
from dj_tools import *
from paper_behavior_functions import *
from ibl_pipeline.analyses import behavior as behavior_analysis

# INITIALIZE A FEW THINGS
seaborn_style()
figpath = figpath()
pal = group_colors()
institution_map, col_names = institution_map()

# ================================= #
# GRAB ALL DATA FROM DATAJOINT
# 3 days before and 3 days after starting biasedChoiceWorld
# ================================= #

use_sessions = query_sessions_around_biased(days_from_trained=[2, 1])
b = subject.Subject * subject.SubjectLab * reference.Lab * \
    acquisition.Session * behavior_analysis.PsychResults & use_sessions[[
        'session_uuid']].to_dict(orient='records')
behav = b.fetch(order_by='institution_short, subject_nickname, session_start_time',
               format='frame').reset_index()
# split the two types of task protocols (remove the pybpod version number
behav['task_protocol'] = behav['task_protocol'].str[14:20]
print(behav.tail(n=10))

# ================================= #
# PREVIOUS CHOICE - SUMMARY PLOT
# ================================= #

fig = sns.FacetGrid(behav, col="task_protocol", 
        hue="previous_choice_name", style="previous_outcome_name", aspect=1, sharex=True, sharey=True)
fig.map(plot_psychometric_summary, "signed_contrast", "choice_right", "subject_nickname").add_legend()
fig.despine(trim=True)
fig.set_titles("{col_name}")
fig._legend.set_title('Previous choice')
fig.set_axis_labels('Signed contrast (%)', 'Rightward choice (%)')
fig.savefig(os.path.join(figpath, "figure4d_history_psychfuncs.pdf"))
fig.savefig(os.path.join(figpath, "figure4d_history_psychfuncs.png"), dpi=600)
print('done')

# ================================= #
# DEFINE HISTORY SHIFT FOR LAG 1
# ================================= #

print('fitting psychometric functions...')
pars = behav.groupby(['institution_code', 'subject_nickname', 'task_protocol',
                      'previous_choice_name', 'previous_outcome_name', ]).apply(fit_psychfunc).reset_index()

shell()

# ================================= #
# STRATEGY SPACE
# ================================= #

fig, ax = plt.subplots(1,1,figsize=[5,5])

# show the shift line for each mouse, per lab
for mouse in biasshift.subject_nickname.unique():
	bs1 = biasshift[biasshift.subject_nickname.str.contains(mouse)]
	bs2 = biasshift_biased[biasshift_biased.subject_nickname.str.contains(mouse)]

	if not bs1.empty and not bs2.empty: # if there is data for this animal in both types of tasks
		ax.plot([bs1.history_postcorrect.item(), bs2.history_postcorrect.item()],
				[bs1.history_posterror.item(), bs2.history_posterror.item()], color='darkgray', ls='-', lw=0.5,
				zorder=-100)

# overlay datapoints for the two task types
sns.scatterplot(x="history_postcorrect", y="history_posterror", style="lab_name",
				color='dimgrey', data=biasshift, ax=ax, legend=False)
sns.scatterplot(x="history_postcorrect", y="history_posterror", style="lab_name",
				color='dodgerblue', data=biasshift_biased, ax=ax, legend=False)

axlim = ceil(np.max([biasshift_biased.history_postcorrect.max(), biasshift_biased.history_posterror.max()]) * 10) / 10

# ax.set_xlim([-axlim,axlim])
# ax.set_ylim([-axlim,axlim])
ax.set_xticks([-axlim,0,axlim])
ax.set_yticks([-axlim,0,axlim])
ax.axhline(linewidth=0.75, color='k', zorder=-500)
ax.axvline(linewidth=0.75, color='k', zorder=-500)

plt.text(axlim/2, axlim/2, 'stay', horizontalalignment='center',verticalalignment='center', style='italic')
plt.text(axlim/2, -axlim/2, 'win stay'+'\n'+'lose switch', horizontalalignment='center',verticalalignment='center', style='italic')
plt.text(-axlim/2, -axlim/2, 'switch', horizontalalignment='center',verticalalignment='center', style='italic')
plt.text(-axlim/2, axlim/2, 'win switch'+'\n'+'lose stay', horizontalalignment='center',verticalalignment='center', style='italic')

ax.set_xlabel("History shift, after correct")
ax.set_ylabel("History shift, after error")

fig.savefig(os.path.join(figpath, "figure4e_historystrategy.pdf"))
fig.savefig(os.path.join(figpath, "figure4e_historystrategy.png"), dpi=600)
plt.close("all")

