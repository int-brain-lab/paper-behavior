#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 29 16:21:33 2019

Stability of performance on easy trials when mice transition from training into biased blocks

@author: Guido Meijer
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from os.path import join, expanduser
import seaborn as sns
from paper_behavior_functions import query_subjects, seaborn_style
import datajoint as dj
from ibl_pipeline import subject, acquisition, action, behavior, reference
from ibl_pipeline.analyses import behavior as behavior_analysis

# Settings
FIG_PATH = join(expanduser('~'), 'Figures', 'Behavior')
SESSION_SHOW = [5, 5]  # Sessions before and after trained to show

# Query list of subjects
subjects = query_subjects(as_dataframe=True)

# Initialize some variables
perf_biased = pd.DataFrame(columns=['mouse', 'perf', 'ses'])
ses_rel = np.concatenate((np.arange(-SESSION_SHOW[0], 0), np.arange(1, SESSION_SHOW[1]+1)))

# Loop over mice
for i, nickname in enumerate(subjects['subject_nickname']):
    if np.mod(i+1, 10) == 0:
        print('Loading data of subject %d of %d' % (i+1, len(subjects)))

    # Get all sessions for this mouse
    ses_date, task = (subject.Subject * acquisition.Session
                      & 'subject_nickname="%s"' % nickname).fetch('session_start_time',
                                                                  'task_protocol')

    # Skip mouse if not enough biased sessions, or if data is not available
    if (len(task) < np.sum(SESSION_SHOW)
            or any(task == None)
            or np.sum(['biased' in s for s in task]) < SESSION_SHOW[1]):
        continue

    # Determine first biased session and get sessions around it
    first_biased = next((i for i, j in enumerate(['biased' in s for s in task]) if j), None)
    ses_ind = np.arange(first_biased-SESSION_SHOW[0], first_biased+SESSION_SHOW[1])

    # Get performance for theses sessions
    for j, ses in enumerate(ses_ind):
        ses_perf = (subject.Subject * behavior_analysis.BehavioralSummaryByDate
                    & 'subject_nickname="%s"' % nickname
                    & 'session_date="%s"' % str(ses_date[ses].date())).fetch('performance_easy')
        perf_biased.loc[len(perf_biased)+1, 'ses'] = ses_rel[j]
        perf_biased.loc[len(perf_biased), 'mouse'] = nickname
        if len(ses_perf) == 1:
            perf_biased.loc[len(perf_biased), 'perf'] = ses_perf[0]*100
        else:
            perf_biased.loc[i*j, 'perf'] = np.nan

# Plot results
perf_biased['ses'] = perf_biased['ses'].astype(float)
perf_biased['perf'] = perf_biased['perf'].astype(float)

f, ax1 = plt.subplots(1, 1, figsize=(5, 5))
seaborn_style()
sns.lineplot(x='ses', y='perf', data=perf_biased, ci=68, ax=ax1)
ax1.set(ylim=[80, 100], xticks=ses_rel, yticks=np.arange(80, 101, 5),
        ylabel='Performance on easy contrasts (%)',
        xlabel='Training sessions relative to transition (days)')
ax1.plot([0, 0], [80, 100], '--r')
ax1.text(-0.5, 90, 'Transition to biased blocks', color='r', rotation=90)
plt.tight_layout(pad=2)

plt.savefig(join(FIG_PATH, 'figure6_performance_to_biased.pdf'), dpi=300)
plt.savefig(join(FIG_PATH, 'figure6_performance_to_biased.png'), dpi=300)
