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
for ax, title in zip(fig.axes.flat, ['50/50', '80/20', '20/80']):
    ax.set_title(title)
    ax.set_xticklabels(['Left', 'Right'], rotation=45)
fig.set_axis_labels('', 'Probability (%)')
fig.savefig(os.path.join(figpath, "figure4_panel_block_distribution.png"), dpi=600)
fig.savefig(os.path.join(figpath, "figure4_panel_block_distribution.pdf"))
plt.close('all')

# ================================= #
# EXAMPLE SESSION TIMECOURSE
# ================================= #

b = (subject.Subject & 'subject_nickname="KS014"') * \
    behavior.TrialSet.Trial * \
    (acquisition.Session & 'task_protocol LIKE "%biased%"' & 'session_start_time BETWEEN "2019-08-30" and "2019-08-31"')
bdat = b.fetch(order_by='session_start_time, trial_id',
               format='frame').reset_index()
behav = dj2pandas(bdat)
assert not behav.empty

# if 100 in df.signed_contrast.values and not 50 in df.signed_contrast.values:
behav['signed_contrast'] = behav['signed_contrast'].replace(-100, -35)
behav['signed_contrast'] = behav['signed_contrast'].replace(100, 35)

for dayidx, behavtmp in behav.groupby(['session_start_time']):

    # 1. patches to show the blocks
    fig, axes = plt.subplots(ncols=1, nrows=1, figsize=(6, 3))
    xmax = min([behavtmp.trial_id.max() + 5, 1005])

    # Loop over data points; create box from errors at each point
    behavtmp['blocks'] = (behavtmp["probabilityLeft"].ne(
        behavtmp["probabilityLeft"].shift()).cumsum())

    for idx, blocknum in behavtmp.groupby('blocks'):
        left = blocknum.trial_id.min()
        width = blocknum.trial_id.max() - blocknum.trial_id.min()
        axes.add_patch(patches.Rectangle((left, 0), width, 100,
                                         fc=cmap_dic[blocknum.probabilityLeft.unique()[
                                             0]],
                                         ec='none', alpha=0.2))

    #%%
    # 2. actual block probabilities as grey line
    behavtmp['stim_sign'] = 100 * ((np.sign(behavtmp.signed_contrast) / 2) + 0.5)
    # sns.scatterplot(x='trial_id', y='stim_sign', data=behav, color='grey',
    #                 marker='o', ax=axes, legend=False, alpha=0.5,
    #                 ec='none', linewidth=0, zorder=2)
    sns.lineplot(x='trial_id', y='stim_sign', color='black', ci=None,
                 data=behavtmp[['trial_id', 'stim_sign']].rolling(10).mean(), ax=axes)
    axes.set(xlim=[-5, xmax], xlabel='Trial number', ylabel='Stimuli on right (%)', ylim=[-1, 101])
    axes.yaxis.label.set_color("black")
    axes.tick_params(axis='y', colors='black')
    #%%

    # 3. ANIMAL CHOICES, rolling window
    rightax = axes.twinx()
    behavtmp['choice_right'] = behavtmp.choice_right * 100
    sns.lineplot(x='trial_id', y='choice_right', color='firebrick', ci=None,
                 data=behavtmp[['trial_id', 'choice_right']].rolling(10).mean(), ax=rightax, linestyle=':')
    rightax.set(xlim=[-5, xmax], xlabel='Trial number',
                ylabel='Rightwards choices (%)', ylim=[-1, 101])
    rightax.yaxis.label.set_color("firebrick")
    rightax.tick_params(axis='y', colors='firebrick')

    axes.set_yticks([0, 50, 100])
    rightax.set_yticks([0, 50, 100])
    axes.set_title('Example session')

    plt.tight_layout()
    fig.savefig(os.path.join(figpath, "figure4_panel_session_course_%s.png" % dayidx), dpi=600)
   # fig.savefig(os.path.join(figpath, "figure4_panel_session_course_%s.pdf" % dayidx))
    plt.close('all')
