"""
SCHEMATIC PSYCHOMETRIC FUNCTIONS
Anne Urai, 2019, CSHL
"""

import pandas as pd
import numpy as np
import sys, os, time
import matplotlib.pyplot as plt
import seaborn as sns
from figure_style import seaborn_style
import datajoint as dj
from IPython import embed as shell # for debugging
from scipy.special import erf # for psychometric functions

## INITIALIZE A FEW THINGS
seaborn_style()
figpath  = os.path.join(os.path.expanduser('~'), 'Data/Figures_IBL')
cmap = sns.diverging_palette(20, 220, n=2, center="dark")

# import wrappers etc
# from ibl_pipeline import reference, subject, action, acquisition, data, behavior
from ibl_pipeline.utils import psychofit as psy

# ================================= #
# SCHEMATIC PSYCHOMETRIC FUNCTIONS
# ================================= #

xvec  = np.arange(-100, 100)
behav = pd.DataFrame({'signed_contrasts':xvec, 'choice':psy.erf_psycho_2gammas([0,30,0,0], xvec)})

behav2 = pd.DataFrame({'signed_contrasts':xvec,	'choice':psy.erf_psycho_2gammas([10,30,0,0], xvec),
                       'prob':np.full_like(xvec, 80)})
behav2 = behav2.append(pd.DataFrame({'signed_contrasts':xvec, 'choice':psy.erf_psycho_2gammas([-10,30,0,0], xvec),
                       'prob':np.full_like(xvec, 20)}))

fig, ax = plt.subplots(1,2, figsize=(4,2), sharex=True, sharey=True)
sns.lineplot(data=behav, x=behav.signed_contrasts, y=100*behav.choice, ax=ax[0], color='k', linewidth=2)

# same in a different panel, for biased blocks
sns.lineplot(x=behav2.signed_contrasts, y=100*behav2.choice, hue=behav2.prob,
             legend=False, ax=ax[1], palette=cmap, linewidth=2)

# ax[1].annotate('Stimulus prior: 80% right', xy=(-10, 70), xytext=(-50, 100),
#                horizontalalignment='center', verticalalignment='top', fontsize=7,
#                arrowprops=dict(facecolor='black', arrowstyle='-|>'))
# # annotate with

plt.tight_layout()
ax[0].set( xlabel='Stimulus contrast (%)', ylabel='Rightward choices (%)', yticks=[0,50,100])
ax[1].set(xlabel='Stimulus contrast (%)', yticks=[0,50,100])

sns.despine(trim=True)
fig.savefig(os.path.join(figpath, "figure1_psychometric.png"), dpi=600)
fig.savefig(os.path.join(figpath, "figure1_psychometric.pdf"))
plt.close('all')