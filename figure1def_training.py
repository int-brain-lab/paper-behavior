"""
TRAINING PROGRESSION FOR AN EXAMPLE MOUSE
Anne Urai, CSHL, 2019
@editor: Gaelle Chapuis, UCL, 2019
"""

import dj_tools
from paper_behavior_functions import *   # noqa
import numpy as np
import pandas as pd
import sys
import matplotlib.pyplot as plt
import seaborn as sns
import datajoint as dj
from IPython import embed as shell     # noqa # for debugging
import os
import datetime
import matplotlib as mpl

# import wrappers etc
from ibl_pipeline import subject, behavior, acquisition
from ibl_pipeline.analyses import behavior as behavioral_analyses
endcriteria = dj.create_virtual_module(
    'SessionEndCriteria', 'group_shared_end_criteria')  # from Miles

# grab some plotting functions from datajoint
sys.path.append(os.path.join(os.path.dirname(__file__),
                             '../IBL-pipeline/prelim_analyses/behavioral_snapshots/'))
import load_mouse_data_datajoint, behavior_plots  # noqa

# INITIALIZE A FEW THINGS
seaborn_style()   # noqa
figpath = figpath()   # noqa
plt.close('all')

# ================================= #
# pick an example mouse
# ================================= #

mouse = 'KS014'
lab = 'cortexlab'

# ==================================================
# CONTRAST HEATMAP
# ================================= #

plt.close('all')
xlims = [pd.Timestamp('2019-08-04T00'), pd.Timestamp('2019-08-31T00')]
fig, ax = plt.subplots(1, 2, figsize=(7, 2.5))
behavior_plots.plot_contrast_heatmap(mouse, lab, ax[0], xlims)
ax[1].axis('off')
ax[0].set_ylabel('Signed contrast (%)')
ax[0].set_xlabel('Training days')
ax[0].set_title('Example mouse')
plt.tight_layout()
fig.savefig(os.path.join(figpath, "figure1_example_contrastheatmap.pdf"))
fig.savefig(os.path.join(figpath, "figure1_example_contrastheatmap.png"), dpi=600)

# ================================================================== #
# PSYCHOMETRIC AND CHRONOMETRIC FUNCTIONS FOR EXAMPLE 3 DAYS
# ================================================================== #

b = ((subject.Subject & 'subject_nickname = "%s"' % mouse) \
    * (subject.SubjectLab & 'lab_name="%s"' % lab) \
    * behavioral_analyses.BehavioralSummaryByDate)
behav = b.fetch(format='frame').reset_index()
behav['training_day'] = behav.training_day - behav.training_day.min() + 1  # start at session 1
days = [2, 7, 10, 14]

for didx, day in enumerate(days):

    # get data for today
    print(day)
    thisdate = behav[behav.training_day == day]['session_date'].dt.strftime('%Y-%m-%d').item()
    b = (subject.Subject & 'subject_nickname = "%s"' % mouse) \
        * (subject.SubjectLab & 'lab_name="%s"' % lab) \
        * (acquisition.Session.proj(session_date='date(session_start_time)') &
           'session_date = "%s"' % thisdate) \
        * behavior.TrialSet.Trial() \
        * endcriteria.SessionEndCriteria()
    behavtmp = dj_tools.dj2pandas(b.fetch(format='frame').reset_index())
    behavtmp['trial_start_time'] = behavtmp.trial_start_time / 60  # in minutes

    # unclear how this can be empty - but if it happens, skip
    if behavtmp.empty:
        continue

    # PSYCHOMETRIC FUNCTIONS
    fig, ax = plt.subplots(1, 1, figsize=(2.5, 2.5))
    behavior_plots.plot_psychometric(behavtmp.rename(
        columns={'signed_contrast': 'signedContrast'}), ax=ax, color='k')
    ax.set(xlabel="Signed contrast (%)", ylim=[0, 1])

    if didx == 0:
        ax.set(ylabel="Rightward choices (%)")
    else:
        ax.set(ylabel=" ")

    ax.set(title='Training day %d' % (day))
    sns.despine(trim=True)
    plt.tight_layout()
    fig.savefig(os.path.join( 
        figpath, "figure1_example_psychfunc_day%d.pdf" % (day)))
    fig.savefig(os.path.join( 
        figpath, "figure1_example_psychfunc_day%d.png" % (day)), dpi=600)

    # # CHRONOMETRIC FUNCTIONS
    # fig, ax = plt.subplots(1, 1, figsize=(2, 2))
    # behavior_plots.plot_chronometric(behavtmp.rename(
    #     columns={'signed_contrast': 'signedContrast'}), ax=ax, color='k')
    # ax.grid(False)
    # ax.set(xlabel="Signed contrast (%)", ylabel="RT (s)", ylim=[0, 1.5])
    # ax.set(title='Day %d' % (didx + 1))

    # # rt axis scaling
    # ax.set(ylim=[0.1, 1.5], yticks=[0.1, 1.5])
    # ax.set_yscale("log")
    # ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda y, pos:
    #                                                       ('{{:.{:1d}f}}'.format(int(np.maximum(
    #                                                           -np.log10(y), 0)))).format(y)))
    # sns.despine(trim=True)
    # plt.tight_layout()
    # fig.savefig(os.path.join(
    #     figpath, "figure1_example_chronfunc_day%d.pdf" % (didx + 1)), dpi=600)

    # ================================================================== #
    # WITHIN-TRIAL DISENGAGEMENT CRITERIA
    # ================================================================== #

    plt.close('all')
    fig, ax = plt.subplots(1, 1, figsize=(3, 2.5))

    # # RTS THROUGHOUT SESSION
    # sns.scatterplot(x='trial_start_time', y='rt', style='correct', hue='correct',
    #                 palette={1: "#1b9e77", 0: "#d95f02"},
    #                 markers={1: 'o', 0: 'X'}, s=10, edgecolors='face',
    #                 alpha=.5, data=behavtmp, ax=ax, legend=False)

    # running median overlaid
    sns.lineplot(x='trial_start_time', y='rt', color='black', ci=None,
                 data=behavtmp[['trial_start_time', 'rt']].rolling(20).median(), ax=ax)
    ax.set(xlabel="Trial number", ylabel="RT (s)", ylim=[0.02, 60])
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda y, pos:
                                                          ('{{:.{:1d}f}}'.format(int(np.maximum(
                                                              -np.log10(y), 0)))).format(y)))
    ax.set(xlabel="Time in session (min)")

    if didx == 0:
        ax.set(ylabel="Trial duration (s)")
    else:
        ax.set(ylabel=" ")
        
    # right y-axis with sliding performance
    # from :
    # https://stackoverflow.com/questions/36988123/pandas-groupby-and-rolling-apply-ignoring-nans

    g1 = behavtmp[['trial_start_time', 'correct_easy']]
    g1['correct_easy'] = g1.correct_easy * 100
    g2 = g1.fillna(0).copy()
    s = g2.rolling(50).sum() / g1.rolling(50).count()  # the actual computation

    ax2 = ax.twinx()
    sns.lineplot(x='trial_start_time', y='correct_easy', color='deepskyblue', ci=None,
                 data=s, ax=ax2)
    ax2.set(xlabel='', 
            ylim=[0, 101], yticks=[0, 50, 100])

    if day == max(days):
        ax2.set(ylabel="Accuracy (%)")
    else:
        ax2.set(ylabel=" ")
        
    ax2.yaxis.label.set_color("deepskyblue")
    ax2.tick_params(axis='y', colors='deepskyblue')
    ax2.spines['right'].set_color('deepskyblue')

    # INDICATE THE REASON AND TRIAL AT WHICH SESSION SHOULD HAVE ENDED
    end_x = behavtmp.loc[behavtmp.trial_id == behavtmp.end_status_index.unique()[
        0], 'trial_start_time'].values.item()
    ax2.axvline(x=end_x, color='darkgrey')
    # ax2.annotate(behavtmp.end_status.unique()[0], xy=(end_x, 100), xytext=(end_x, 105),
    #              arrowprops={'arrowstyle': "->", 'connectionstyle': "arc3"})
    print(behavtmp.end_status.unique()[0])

    ax.set(title='Training day %d' % (day))
    # sns.despine(trim=True)
    sns.despine(ax=ax, top=True, left=False, right=False)
    sns.despine(ax=ax2, top=True,  left=False, right=False)

    #sns.despine(ax=ax2, right=True, left=False)

    plt.tight_layout()
    fig.savefig(os.path.join(
        figpath, "figure1_example_disengagement_day%d.pdf" % (day)))
    fig.savefig(os.path.join(
        figpath, "figure1_example_disengagement_day%d.png" % (day)), dpi=600)

    print(didx)
    print(thisdate)
