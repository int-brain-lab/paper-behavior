"""
SCHEMATIC OF BLOCK STRUCTURE
Anne Urai, CSHL, 2019
"""

import pandas as pd
import os
import seaborn as sns
from figure_style import seaborn_style

# INITIALIZE A FEW THINGS
seaborn_style()
figpath = os.path.join(os.path.expanduser('~'), 'Data/Figures_IBL')
cmap = sns.diverging_palette(20, 220, n=3, center="dark")

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
fig.set_axis_labels('Stimulus side', 'Probability (%)')
fig.savefig(os.path.join(
    figpath, "figure1_panel_block_distribution.png"), dpi=600)
fig.savefig(os.path.join(figpath, "figure1_panel_block_distribution.pdf"))
