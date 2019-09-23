"""
CONTRAST HEATMAP FOR AN EXAMPLE MOUSE
Anne Urai, CSHL, 2019
"""

import pandas as pd
import numpy as np
import sys
import matplotlib.pyplot as plt
import seaborn as sns
from paper_behavior_functions import *
import datajoint as dj
from IPython import embed as shell  # for debugging
import  re, datetime, os, glob
from datetime import timedelta
import matplotlib as mpl

# import wrappers etc
from ibl_pipeline import reference, subject, action, acquisition, data, behavior
from ibl_pipeline.utils import psychofit as psy
from ibl_pipeline.analyses import behavior as behavioral_analyses

sys.path.append("../IBL-pipeline")
from IBL-pipeline.prelim_analyses import behavior_plots, load_mouse_data_datajoint

# INITIALIZE A FEW THINGS
seaborn_style()
figpath = figpath()
plt.close('all')


# ================================= #
# pick an example mouse
# ================================= #

mouse = 'CSHL_015'
lab = 'churchlandlab'

weight_water, baseline = load_mouse_data_datajoint.get_water_weight(mouse, lab)
xlims = [weight_water.date.min() - timedelta(days=2), weight_water.date.max() + timedelta(days=2)]

# ================================= #
# CONTRAST HEATMAP
# ================================= #

fig, ax = plt.subplots(1, 2, figsize=(7, 2.5))
behavior_plots.plot_contrast_heatmap(mouse, lab, ax[0], xlims)
ax[1].axis('off')
ax[0].set_ylabel('Signed contrast (%)')
ax[0].set_xlabel('Training days')        
ax[0].set_title('Mouse %s' % mouse)
plt.tight_layout()
fig.savefig(os.path.join(figpath, "figure3_example_contrastheatmap.png"))

# ================================================================== #
# PSYCHOMETRIC AND CHRONOMETRIC FUNCTIONS FOR EXAMPLE 3 DAYS
# ================================================================== #

behav = load_mouse_data_datajoint.get_behavior(mouse, lab)
days = [3, 10, 27]

for didx, day in enumerate(days):

    behavtmp = behav.loc[behav['days'] == day, :].copy()
    
    # PSYCHOMETRIC FUNCTIONS
    fig, ax = plt.subplots(1, 1, figsize=(2, 2))
    behavior_plots.plot_psychometric(behavtmp, ax=ax, color='k')
    ax.set(xlabel="Signed contrast (%)", ylabel="Rightward choices (%)", ylim=[0, 1])
    ax.set(title=pd.to_datetime(behavtmp['start_time'].unique()[0]).strftime('%b-%d'))
    sns.despine(trim=True)
    plt.tight_layout()
    fig.savefig(os.path.join(figpath, "figure3_example_psychfunc_day%d.png" % day))

    # CHRONOMETRIC FUNCTIONS
    fig, ax = plt.subplots(1, 1, figsize=(2, 2))
    behavior_plots.plot_chronometric(behavtmp, ax=ax, color='k')
    ax.grid(False)
    ax.set(xlabel="Signed contrast (%)", ylabel="RT (s)", ylim=[0, 1.5])
    ax.set(title=pd.to_datetime(behavtmp['start_time'].unique()[0]).strftime('%b-%d'))

    # RT SCALING
    ax.set(ylim=[0.1, 1.5], yticks=[0.1, 1.5])
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda y, pos:
        ('{{:.{:1d}f}}'.format(int(np.maximum(-np.log10(y), 0)))).format(y)))

    sns.despine(trim=True)
    plt.tight_layout()
    fig.savefig(os.path.join(figpath, "figure3_example_chronfunc_day%d.png" % day))
