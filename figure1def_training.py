"""
Training progression for an example mouse

@author: Anne Urai, Gaelle Chapuis, Miles Wells
21 April 2020
"""
import os

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import datajoint as dj

# import wrappers etc
from paper_behavior_functions import EXAMPLE_MOUSE
from ibl_pipeline import subject, behavior, acquisition
from ibl_pipeline.analyses import behavior as behavioral_analyses
endcriteria = dj.create_virtual_module(
    'SessionEndCriteriaImplemented', 'group_shared_end_criteria')

# grab some plotting functions from datajoint
# (this is a tricky dependency, as is it can not be run in a python shell, it makes the whole file
# need to run as an executable eg. >>> python figure1_training.py in windows command prompt)
# import sys
# sys.path.append(os.path.join(os.path.dirname(__file__),
#                              '../IBL-pipeline/prelim_analyses/behavioral_snapshots/'))
# import ibl_pipeline.prelim_analyses.behavioral_snapshots.behavior_plots  # noqa

# this only works if conda develop ./IBL-pipeline/prelim_analyses/behavioral_snapshots/ has been added to iblenv
import behavior_plots
import dj_tools
from paper_behavior_functions import seaborn_style, figpath, \
    FIGURE_HEIGHT, FIGURE_WIDTH

# ================================= #
# INITIALIZE A FEW THINGS
# ================================= #

seaborn_style()  # noqa
figpath = figpath()  # noqa
plt.close('all')
# FIGURE_WIDTH = 6 # make narrower

# ================================= #
# Get lab name of example mouse
# ================================= #

lab = (subject.SubjectLab * subject.Subject & 'subject_nickname = "%s"' % EXAMPLE_MOUSE) \
      .fetch1('lab_name')
days = [2, 7, 10, 14]

# ==================================================
# CONTRAST HEATMAP
# ================================= #

plt.close('all')
fig, ax = plt.subplots(1, 2, figsize=(FIGURE_WIDTH / 2, FIGURE_HEIGHT))
ax[1].axis('off')
xlims = [pd.Timestamp('2019-08-04T00'), pd.Timestamp('2019-08-31T00')]
behavior_plots.plot_contrast_heatmap(EXAMPLE_MOUSE, lab, ax[0], xlims)
ax[0].set(ylabel='Contrast (%)', xlabel='Training day',
          xticks=[d + 1.5 for d in days], xticklabels=days,
          yticklabels=['-100', '-50', '-25', '-12.5', '-6.25', '0',
                       '6.25', '12.5', '25', '50', '100'])
for item in ax[0].get_xticklabels():
    item.set_rotation(-0)
plt.tight_layout()
fig.savefig(os.path.join(figpath, "figure1_example_contrastheatmap.pdf"))
fig.savefig(os.path.join(
    figpath, "figure1_example_contrastheatmap.png"), dpi=600)

# ================================================================== #
# PSYCHOMETRIC AND CHRONOMETRIC FUNCTIONS FOR EXAMPLE 3 DAYS
# ================================================================== #

# make these a bit more narrow

b = ((subject.Subject & 'subject_nickname = "%s"' % EXAMPLE_MOUSE)
     * (subject.SubjectLab & 'lab_name="%s"' % lab)
     * behavioral_analyses.BehavioralSummaryByDate)
behav = b.fetch(format='frame').reset_index()
behav['training_day'] = behav.training_day - \
    behav.training_day.min() + 1  # start at session 1

for didx, day in enumerate(days):

    # get data for today
    print(day)
    thisdate = behav[behav.training_day ==
                     day]['session_date'].dt.strftime('%Y-%m-%d').item()
    b = (subject.Subject & 'subject_nickname = "%s"' % EXAMPLE_MOUSE) \
        * (subject.SubjectLab & 'lab_name="%s"' % lab) \
        * (acquisition.Session.proj(session_date='date(session_start_time)') &
           'session_date = "%s"' % thisdate) \
        * behavior.TrialSet.Trial() \
        * endcriteria.SessionEndCriteriaImplemented()
    behavtmp = dj_tools.dj2pandas(b.fetch(format='frame').reset_index())
    behavtmp['trial_start_time'] = behavtmp.trial_start_time / 60  # in minutes

    # unclear how this can be empty - but if it happens, skip
    if behavtmp.empty:
        continue

    # PSYCHOMETRIC FUNCTIONS
    fig, ax = plt.subplots(1, 1, figsize=(FIGURE_WIDTH/5, FIGURE_HEIGHT*0.9))
    dj_tools.plot_psychometric(behavtmp.signed_contrast,
                               behavtmp.choice_right,
                               behavtmp.trial_id,
                               ax=ax, color='k')
    ax.set(xlabel="Contrast (%)")

    if didx == 0:
        ax.set(ylabel="Rightward choices (%)")
    else:
        ax.set(ylabel=" ", yticklabels=[])

    # ax.set(title='Training day %d' % (day))
    sns.despine(trim=True)
    plt.tight_layout()
    fig.savefig(os.path.join(
        figpath, "figure1_example_psychfunc_day%d.pdf" % (day)))
    fig.savefig(os.path.join(
        figpath, "figure1_example_psychfunc_day%d.png" % (day)), dpi=600)

    # ================================================================== #
    # WITHIN-TRIAL DISENGAGEMENT CRITERIA
    # ================================================================== #

    plt.close('all')
    fig, ax = plt.subplots(2, 1, sharex=True,
                           figsize=(FIGURE_WIDTH/5, FIGURE_HEIGHT*1.5))

    # running median overlaid
    sns.lineplot(x='trial_start_time', y='rt', color='black', ci=None,
                 data=behavtmp[['trial_start_time', 'rt']].rolling(20).median(), ax=ax[0])
    ax[0].set(xlabel="", ylabel="RT (s)", ylim=[0.1, 20])
    ax[0].set_yscale("log")

    # fix xlims
    if didx == 0:
        xlim = [0, 60]
    elif didx == 1:
        xlim = [0, 80]
    elif didx == 2:
        xlim = [0, 60]
    elif didx == 3:
        xlim = [0, 60]

    ax[0].set(yticks=[0.1, 1, 10, 20],
              yticklabels=['0.1', '1', '10', ''], xlim=xlim)

    if didx == 0:
        ax[0].set(ylabel="Trial duration (s)")
    else:
        ax[0].set(ylabel=" ", yticklabels=[])

    # right y-axis with sliding performance
    # from :
    # https://stackoverflow.com/questions/36988123/pandas-groupby-and-rolling-apply-ignoring-nans

    g1 = behavtmp[['trial_start_time', 'correct_easy']].copy()
    g1['correct_easy'] = g1.correct_easy * 100
    g2 = g1.fillna(0).copy()
    s = g2.rolling(50).sum() / g1.rolling(50).count()  # the actual computation

    sns.lineplot(x='trial_start_time', y='correct_easy', color='black', ci=None,
                 data=s, ax=ax[1])

    if day == min(days):
        ax[1].set(ylabel="Accuracy (%)")
    else:
        ax[1].set(ylabel=" ", yticklabels=[])

    ax[1].set(xlabel='Time (min)', ylim=[25, 110], yticks=[25, 50, 75, 100],
              xlim=ax[0].get_xlim(), xticks=[0, 20, 40, 60, 80])

    # INDICATE THE REASON AND TRIAL AT WHICH SESSION SHOULD HAVE ENDED
    idx = behavtmp.trial_id == behavtmp.end_status_index.unique()[0]
    end_x = behavtmp.loc[idx, 'trial_start_time'].values.item()
    ax[0].axvline(x=end_x, color='darkgrey', linestyle=':')
    ax[1].axvline(x=end_x, color='darkgrey', linestyle=':')
    # ax2.annotate(behavtmp.end_status.unique()[0], xy=(end_x, 100), xytext=(end_x, 105),
    #              arrowprops={'arrowstyle': "->", 'connectionstyle': "arc3"})
    print(behavtmp.end_status.unique()[0])

    ax[0].set(title='Day %d' % day)
    sns.despine(trim=True)
    plt.tight_layout(h_pad=-0.05)
    fig.savefig(os.path.join(
        figpath, "figure1_example_disengagement_day%d.pdf" % day))
    fig.savefig(os.path.join(
        figpath, "figure1_example_disengagement_day%d.png" % day), dpi=600)

    print(didx)
    print(thisdate)
