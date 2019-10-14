"""
PSYCHOMETRIC AND CHRONOMETRIC FUNCTIONS IN BIASED BLOCKS
Anne Urai, CSHL, 2019
"""

from dj_tools import *
import pandas as pd
import numpy as np
import sys
import os
import matplotlib.pyplot as plt
import seaborn as sns
from paper_behavior_functions import *
import datajoint as dj
from IPython import embed as shell  # for debugging
from scipy.special import erf  # for psychometric functions

# import wrappers etc
from ibl_pipeline import reference, subject, action, acquisition, data, behavior
from ibl_pipeline.utils import psychofit as psy
from ibl_pipeline.analyses import behavior as behavioral_analyses

sys.path.insert(0, '../python')

# INITIALIZE A FEW THINGS
sns.set(style="ticks", context="paper", font_scale=1.2)
figpath = figpath()
cmap = sns.diverging_palette(20, 220, n=3, center="dark")
sns.set_palette(cmap)  # palette for water types
pal = sns.color_palette("colorblind", 7)

# ================================= #
# GET DATA FROM TRAINED ANIMALS
# ================================= #

# TODO: WAIT FOR SHAN TO ADD training_day  AND COMPLETE THE QUERY FOR THE RIGHT SESSIONS
behav = query_sessions_around_biased(days_from_trained=[10, 10])
assert(~behav.empty)
print(behav.describe())

# give each session a number
behav['session_day'] = behav.bias
for index, group in behav.groupby(['subject_nickname']):
      behav['session_day'][behav.index.isin(group.index)] = (group['session_date'] - group['session_date'].min()).dt.days

behav = behav.loc[behav['session_day'] <= 20]

# same in a different panel, for biased blocks
fig, ax = plt.subplots(1, 1, figsize=(5, 2.5), sharex=True, sharey=True)
sns.lineplot(data=behav, x="session_day",  y="bias", hue="prob_left", 
    ax=ax, palette=cmap, legend=False, err_style="bars", marker='o')
ax.set(xlabel='Days', ylabel='Choice bias')
sns.despine(trim=True)
plt.tight_layout()
fig.savefig(os.path.join(figpath, "figure4c_bias_stability.pdf"))
fig.savefig(os.path.join(figpath, "figure4c_bias_stability.png"), dpi=600)
plt.close('all')
