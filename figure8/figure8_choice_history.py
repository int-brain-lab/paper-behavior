"""
Plot full psychometric functions as a function of choice history,
and separately for 20/80 and 80/20 blocks
"""

import pandas as pd
import numpy as np
import sys, os, time
import matplotlib.pyplot as plt
import seaborn as sns
import datajoint as dj
from IPython import embed as shell # for debugging
from math import ceil
from figure_style import seaborn_style

# import wrappers etc
from ibl_pipeline import reference, subject, action, acquisition, data, behavior
from ibl_pipeline.utils import psychofit as psy

sys.path.insert(0, '/Users/urai/Documents/code/analysis_IBL/python')
from dj_tools import *

plt.close("all")
seaborn_style()
figpath  = os.path.join(os.path.expanduser('~'), 'Data/Figures_IBL')

# ================================= #
# GRAB ALL DATA FROM DATAJOINT
# only data when the animal is trained,
# but hasn't seen biased blocks yet
# ================================= #

use_subjects = subject.Subject * subject.SubjectLab * subject.SubjectProject & 'subject_project="ibl_neuropixel_brainwide_01"'
# criterion = behavioral_analyses.SessionTrainingStatus()
# sess = ((acquisition.Session & 'task_protocol LIKE "%trainingChoiceWorld%"') \
# 		* (criterion & 'training_status="trained"')) * use_subjects

# take all trials that include 0% contrast, instead of those where the animal is trained
sess = (acquisition.Session & (behavior.TrialSet.Trial() & 'ABS(trial_stim_contrast_left-0)<0.0001' \
	& 'ABS(trial_stim_contrast_right-0)<0.0001') & 'task_protocol like "%trainingChoiceWorld%"') \
	* use_subjects

b 		= (behavior.TrialSet.Trial & sess) * subject.Subject() * subject.SubjectLab()
bdat 	= pd.DataFrame(b.fetch(order_by='subject_nickname, session_start_time, trial_id'))
behav 	= dj2pandas(bdat)
print(behav.tail(n=10))
print(behav.groupby(['previous_outcome_name'])['repeat'].mean())

# PREVIOUS CHOICE - SUMMARY PLOT
fig = sns.FacetGrid(behav, col="previous_outcome_name", hue="previous_choice_name", aspect=1, sharex=True, sharey=True)
fig.map(plot_psychometric, "signed_contrast", "choice_right", "subject_nickname").add_legend()
fig.despine(trim=True)
fig.set_titles("{col_name}")
fig._legend.set_title('Previous choice')
fig.set_axis_labels('Signed contrast (%)', 'Rightward choice (%)')
fig.savefig(os.path.join(figpath, "history_trainingChoiceWorld.pdf"))
fig.savefig(os.path.join(figpath, "history_trainingChoiceWorld.png"), dpi=600)
print('done')

# ================================= #
# REPEAT FOR BIASEDCHOICEWORLD DATA
# this should shift their history preferences
# ================================= #

sess = (acquisition.Session & (behavior.TrialSet.Trial() & 'ABS(trial_stim_contrast_left-0)<0.0001' \
	& 'ABS(trial_stim_contrast_right-0)<0.0001') & 'task_protocol like "%biasedChoiceWorld%"') * use_subjects
b 				= (behavior.TrialSet.Trial & sess) * subject.Subject() * subject.SubjectLab()
bdat 			= pd.DataFrame(b.fetch(order_by='subject_nickname, session_start_time, trial_id'))
behav_biased 	= dj2pandas(bdat)
print(behav_biased.tail(n=10))
print(behav_biased.groupby(['previous_outcome_name'])['repeat'].mean())

# PREVIOUS CHOICE - SUMMARY PLOT
fig = sns.FacetGrid(behav_biased, col="previous_outcome_name", hue="previous_choice_name",
					aspect=1, sharex=True, sharey=True)
fig.map(plot_psychometric, "signed_contrast", "choice_right", "subject_nickname").add_legend()
fig.despine(trim=True)
fig.set_titles("{col_name}")
fig._legend.set_title('Previous choice')
fig.set_axis_labels('Signed contrast (%)', 'Rightward choice (%)')
fig.savefig(os.path.join(figpath, "history_biasedChoiceWorld.pdf"))
fig.savefig(os.path.join(figpath, "history_biasedChoiceWorld.png"), dpi=600)
print('done')

# ================================= #
# DEFINE HISTORY SHIFT FOR LAG 1
# ================================= #

def compute_biasshift_posterror(x):

	xax = np.arange(-100, 100)
	x_right = x.loc[np.isclose(x['previous_choice'], 1) & np.isclose(x['previous_outcome'], -1)]
	x_left = x.loc[np.isclose(x['previous_choice'], -1) & np.isclose(x['previous_outcome'], -1)]

	y_right = psy.erf_psycho_2gammas([x_right['bias'].item(),
		x_right['threshold'].item(), x_right['lapselow'].item(), x_right['lapsehigh'].item()], xax)
	y_left = psy.erf_psycho_2gammas([x_left['bias'].item(),
		x_left['threshold'].item(), x_left['lapselow'].item(), x_left['lapsehigh'].item()], xax)

	shift_posterror = (y_right[xax == 0] - y_left[xax == 0]).item()
	return shift_posterror

def compute_biasshift_postcorrect(x):

	xax = np.arange(-100, 100)
	x_right = x.loc[np.isclose(x['previous_choice'], 1) & np.isclose(x['previous_outcome'], 1)]
	x_left = x.loc[np.isclose(x['previous_choice'], -1) & np.isclose(x['previous_outcome'], 1)]

	y_right = psy.erf_psycho_2gammas([x_right['bias'].item(),
		x_right['threshold'].item(), x_right['lapselow'].item(), x_right['lapsehigh'].item()], xax)
	y_left = psy.erf_psycho_2gammas([x_left['bias'].item(),
		x_left['threshold'].item(), x_left['lapselow'].item(), x_left['lapsehigh'].item()], xax)

	shift_postcorrect = (y_right[xax == 0] - y_left[xax == 0])
	return shift_postcorrect[0]

# ================================= #
# COMPUTE HISTORY SHIFT - TRAININGCHOICEWORLD
# ================================= #

print('fitting psychometrics...')
pars = behav.groupby(['lab_name', 'subject_nickname', 'previous_choice', 'previous_outcome']).apply(fit_psychfunc).reset_index()

# check if these fits worked as expected
# parameters should be within reasonable bounds...
assert pars['lapselow'].mean() < 0.4
assert pars['lapsehigh'].mean() < 0.4

# compute a 'bias shift' per animal
biasshift = pars.groupby(['lab_name', 'subject_nickname']).apply(compute_biasshift_postcorrect).reset_index()
biasshift = biasshift.rename(columns={0: 'history_postcorrect'})

biasshift2 = pars.groupby(['lab_name', 'subject_nickname']).apply(compute_biasshift_posterror).reset_index()
biasshift2 = biasshift2.rename(columns={0: 'history_posterror'})
biasshift['history_posterror'] = biasshift2.history_posterror.copy()

# ================================= #
# COMPUTE HISTORY SHIFT - TRAININGCHOICEWORLD
# ================================= #

print('fitting psychometrics...')
pars = behav_biased.groupby(['lab_name', 'subject_nickname', 'previous_choice', 'previous_outcome']).apply(fit_psychfunc).reset_index()

# check if these fits worked as expected
# parameters should be within reasonable bounds...
assert pars['lapselow'].mean() < 0.4
assert pars['lapsehigh'].mean() < 0.4

# compute a 'bias shift' per animal
biasshift_biased = pars.groupby(['lab_name', 'subject_nickname']).apply(compute_biasshift_postcorrect).reset_index()
biasshift_biased = biasshift_biased.rename(columns={0: 'history_postcorrect'})

biasshift_biased2 = pars.groupby(['lab_name', 'subject_nickname']).apply(compute_biasshift_posterror).reset_index()
biasshift_biased2 = biasshift_biased2.rename(columns={0: 'history_posterror'})
biasshift_biased['history_posterror'] = biasshift_biased2.history_posterror.copy()

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

ax.set_xlim([-axlim,axlim])
ax.set_ylim([-axlim,axlim])
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

fig.savefig(os.path.join(figpath, "history_strategy.pdf"))
fig.savefig(os.path.join(figpath, "history_strategy.png"), dpi=600)
plt.close("all")