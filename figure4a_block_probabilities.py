"""
SCHEMATIC OF BLOCK STRUCTURE
Anne Urai, CSHL, 2019
"""

from IPython import embed as shell  # for debugging
from ibl_pipeline import reference, subject, action, acquisition, data, behavior
import pandas as pd
import os
import seaborn as sns
import numpy as np
from paper_behavior_functions import *
from dj_tools import *
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# INITIALIZE A FEW THINGS
seaborn_style()
figpath = figpath()
cmap = sns.diverging_palette(20, 220, n=3, center="dark")
cmap_dic = {20: cmap[0], 50: cmap[1], 80: cmap[2]}
# ================================= #
# SCHEMATIC OF THE BLOCKS
# ================================= #

# behav = pd.DataFrame({'probability_left': [50, 50, 20, 20, 80, 80],
#                       'stimulus_side': [-1, 1, -1, 1, -1, 1],
#                       'prob': [50, 50, 20, 80, 80, 20]})

# fig = sns.FacetGrid(behav,
#                     col="probability_left", hue="probability_left", col_wrap=3, col_order=[50, 20, 80],
#                     palette=cmap, sharex=True, sharey=True, aspect=0.6, height=2.2)
# # fig.map(sns.distplot, "stimulus_side", kde=False, norm_hist=True, bins=2, hist_kws={'rwidth':1})
# fig.map(sns.barplot, "stimulus_side", "prob")
# fig.set(xticks=[-0, 1], xlim=[-0.5, 1.5],
#         ylim=[0, 100], yticks=[0, 50, 100], yticklabels=[])
# for ax, title in zip(fig.axes.flat, ['50/50', '20/80', '80/20']):
#     ax.set_title(title)
#     ax.set_xticklabels(['Left', 'Right'], rotation=45)
# fig.set_axis_labels('', 'Probability (%)')
# fig.savefig(os.path.join(figpath, "figure4_panel_block_distribution.png"), dpi=600)
# fig.savefig(os.path.join(figpath, "figure4_panel_block_distribution.pdf"))
# plt.close('all')

# ================================= #
# EXAMPLE SESSION TIMECOURSE
# ================================= #

b = (subject.Subject & 'subject_nickname="ibl_witten_06"') * \
    behavior.TrialSet.Trial * \
    (acquisition.Session & 'session_start_time BETWEEN "2019-04-15" AND "2019-04-16"')
bdat = b.fetch(order_by='session_start_time, trial_id',
               format='frame').reset_index()
behav = dj2pandas(bdat)
behav = behav.loc[behav['trial_id'] <= 1000]

# if 100 in df.signed_contrast.values and not 50 in df.signed_contrast.values:
behav['signed_contrast'] = behav['signed_contrast'].replace(-100, -35)
behav['signed_contrast'] = behav['signed_contrast'].replace(100, 35)

# 1. patches to show the blocks
fig, axes = plt.subplots(ncols=1, nrows=1, figsize=(6, 3))

# Loop over data points; create box from errors at each point
behav['blocks'] = (behav["probabilityLeft"].ne(
    behav["probabilityLeft"].shift()).cumsum())

for idx, blocknum in behav.groupby('blocks'):
    left = blocknum.trial_id.min()
    width = blocknum.trial_id.max() - blocknum.trial_id.min()
    axes.add_patch(patches.Rectangle((left, -36), width, 72,
                                     fc=cmap_dic[blocknum.probabilityLeft.unique()[
                                         0]],
                                     ec='none', alpha=0.3))

# 2. stimuli as grey dots
sns.scatterplot(x='trial_id', y='signed_contrast', data=behav, color='grey',
                marker='o', ax=axes, legend=False, alpha=0.5,
                ec='none', linewidth=0, zorder=2)

# 3. ANIMAL CHOICES, rolling window
rightax = axes.twinx()
behav['choice_right'] = behav.choice_right * 100
sns.lineplot(x='trial_id', y='choice_right', color='black', ci=None,
             data=behav[['trial_id', 'choice_right']].rolling(10).mean(), ax=rightax)
axes.set(xlim=[-5, 1005], xlabel='Trial number', ylabel='Signed contrast (%)', ylim=[-37, 37])
axes.yaxis.label.set_color("grey")
axes.tick_params(axis='y', colors='grey')
rightax.set(xlim=[-5, 1005], xlabel='Trial number',
            ylabel='Rightwards choices (%)', ylim=[-1, 101])

# SAVE
axes.set_yticks([-35, -25, -12.5, 0, 12.5, 25, 35])
axes.set_yticklabels(['-100', '-25', '-12.5', '0', '12.5', '25', '100'],
                     size='small')
rightax.set_yticks([0, 50, 100])
axes.set_title('Example session')

plt.tight_layout()
fig.savefig(os.path.join(figpath, "figure4_panel_session_course.png"), dpi=600)
fig.savefig(os.path.join(figpath, "figure4_panel_session_course.pdf"))
plt.close('all')
