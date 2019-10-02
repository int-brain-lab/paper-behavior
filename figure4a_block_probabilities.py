"""
SCHEMATIC OF BLOCK STRUCTURE
Anne Urai, CSHL, 2019
"""

import pandas as pd
import os
import seaborn as sns
import numpy as np
from paper_behavior_functions import *
from dj_tools import *

# INITIALIZE A FEW THINGS
seaborn_style()
figpath = figpath()
cmap = sns.diverging_palette(20, 220, n=3, center="dark")
from ibl_pipeline import reference, subject, action, acquisition, data, behavior
from IPython import embed as shell  # for debugging

# ================================= #
# SCHEMATIC OF THE BLOCKS
# ================================= #

behav = pd.DataFrame({'probability_left': [50, 50, 20, 20, 80, 80],
                      'stimulus_side': [-1, 1, -1, 1, -1, 1],
                      'prob': [50, 50, 20, 80, 80, 20]})

fig = sns.FacetGrid(behav,
                    col="probability_left", hue="probability_left", col_wrap=3, col_order=[50, 20, 80],
                    palette=cmap, sharex=True, sharey=True, aspect=0.6, height=2.2)
# fig.map(sns.distplot, "stimulus_side", kde=False, norm_hist=True, bins=2, hist_kws={'rwidth':1})
fig.map(sns.barplot, "stimulus_side", "prob")
fig.set(xticks=[-0, 1], xlim=[-0.5, 1.5],
        ylim=[0, 100], yticks=[0, 50, 100], yticklabels=[])
for ax, title in zip(fig.axes.flat, ['50/50', '20/80', '80/20']):
    ax.set_title(title)
    ax.set_xticklabels(['Left', 'Right'], rotation=45)
fig.set_axis_labels('', 'Probability (%)')
fig.savefig(os.path.join(figpath, "figure4_panel_block_distribution.png"), dpi=600)
fig.savefig(os.path.join(figpath, "figure4_panel_block_distribution.pdf"))
plt.close('all')

# ================================= #
# EXAMPLE SESSION TIMECOURSE
# ================================= #

b = (subject.Subject & 'subject_nickname="ibl_witten_06"') * \
    behavior.TrialSet.Trial * (acquisition.Session & 'session_start_time BETWEEN "2019-04-15" AND "2019-04-16"')
bdat = b.fetch(order_by='session_start_time, trial_id', format='frame').reset_index()
behav = dj2pandas(bdat)
behav = behav.loc[behav['trial_id'] <= 1000]

fig, axes = plt.subplots(ncols=1, nrows=1, figsize=(6, 3))
rightax = axes.twinx()
sns.lineplot(x=behav.trial_id, y=behav.probabilityLeft, color='k', ax=rightax, zorder=1)
sns.scatterplot(x=behav.trial_id, y=behav.signed_contrast, hue=np.sign(behav.signed_contrast), 
                palette=cmap, marker='.', ax=axes, legend=False, alpha=0.5, 
                edgecolors='face', zorder=2)
# fig.set_titles("")
axes.set(xlim=[-5, 1005], xlabel='Trial number', ylabel='Signed contrast (%)') 
rightax.set(xlim=[-5,1005], ylabel='P(Left)', yticks=[20,50,80])
plt.tight_layout()
fig.savefig(os.path.join(figpath, "figure4_panel_session_course.png"), dpi=600)
fig.savefig(os.path.join(figpath, "figure4_panel_session_course.pdf"))
plt.close('all')
