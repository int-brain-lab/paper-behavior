"""
SCHEMATIC PSYCHOMETRIC FUNCTIONS
Anne Urai, 2019, CSHL
"""

from ibl_pipeline.utils import psychofit as psy
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from paper_behavior_functions import *
from IPython import embed as shell


# INITIALIZE A FEW THINGS
seaborn_style()
cmap = sns.diverging_palette(20, 220, n=3, center="dark")
figpath = figpath()  # grab this path

# ================================= #
# SCHEMATIC PSYCHOMETRIC FUNCTIONS
# ================================= #

xvec = np.arange(-100, 100)
behav = pd.DataFrame({'signed_contrasts': xvec,
                      'choice': psy.erf_psycho_2gammas([0, 30, 0, 0], xvec), 
                      'prob': np.full_like(xvec, 50)})

behav2 = behav.copy()
behav2 = behav2.append(pd.DataFrame({'signed_contrasts': xvec,
                       'choice': psy.erf_psycho_2gammas([15, 30, 0, 0], xvec),
                       'prob': np.full_like(xvec, 80)}))
behav2 = behav2.append(pd.DataFrame({'signed_contrasts': xvec,
                                     'choice': psy.erf_psycho_2gammas([-15, 30, 0, 0], xvec),
                                     'prob': np.full_like(xvec, 20)}))

fig, ax = plt.subplots(1, 1, figsize=(2.5, 2.5), sharex=True, sharey=True)
sns.lineplot(data=behav, x=behav.signed_contrasts, y=100 *
             behav.choice, ax=ax, color='k', linewidth=2)
ax.set(xlabel='Stimulus contrast (%)',
          ylabel='Rightward choices (%)', yticks=[0, 50, 100])
sns.despine(trim=True)
plt.tight_layout()
fig.savefig(os.path.join(figpath, "figure1_psychometric.png"), dpi=600)
fig.savefig(os.path.join(figpath, "figure1_psychometric.pdf"))
plt.close('all')

# same in a different panel, for biased blocks
fig, ax = plt.subplots(1, 1, figsize=(2.5, 2.5), sharex=True, sharey=True)
sns.lineplot(data=behav2, x=behav2.signed_contrasts, y=100 *
             behav2.choice, ax=ax, linewidth=2, hue=behav2.prob, palette=cmap, legend=False)

ax.annotate('80/20', xy=(-20, 50), xytext=(-40, 80), color=cmap[0], fontsize=12,
    arrowprops=dict(facecolor=cmap[0], shrink=0.05), ha='right')
ax.annotate('20/80', xy=(20, 50), xytext=(40, 20), color=cmap[2], fontsize=12,
    arrowprops=dict(facecolor=cmap[2], shrink=0.05))

plt.tight_layout()
ax.set(xlabel='Stimulus contrast (%)',
       ylabel='Rightward choices (%)', yticks=[0, 50, 100])
sns.despine(trim=True)
fig.savefig(os.path.join(figpath, "figure4a_psychometric.png"), dpi=600)
fig.savefig(os.path.join(figpath, "figure4a_psychometric.pdf"))
plt.close('all')
