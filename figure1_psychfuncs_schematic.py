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

fig, ax = plt.subplots(1, 2, figsize=(5, 2.5), sharex=True, sharey=True)
sns.lineplot(data=behav, x=behav.signed_contrasts, y=100 *
             behav.choice, ax=ax[0], color='k', linewidth=2)

# same in a different panel, for biased blocks
sns.lineplot(x=behav2.signed_contrasts, y=100 * behav2.choice, hue=behav2.prob,
             legend=False, ax=ax[1], palette=cmap, linewidth=2)

plt.tight_layout()
ax[0].set(xlabel='Stimulus contrast (%)',
          ylabel='Rightward choices (%)', yticks=[0, 50, 100])
ax[1].set(xlabel='Stimulus contrast (%)', ylabel='Rightward choices (%)', yticks=[0, 50, 100])

sns.despine(trim=True)
fig.savefig(os.path.join(figpath, "figure1_psychometric.png"), dpi=600)
fig.savefig(os.path.join(figpath, "figure1_psychometric.pdf"))
plt.close('all')
