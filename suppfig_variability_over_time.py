#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Quantify the variability of behavioral metrics within and between labs of mouse behavior.
This script doesn't perform any analysis but plots summary statistics over labs.

Guido Meijer, Miles Wells
16 Jan 2020
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from os.path import join
from paper_behavior_functions import (seaborn_style, institution_map, group_colors, figpath,
                                      query_subjects, FIGURE_WIDTH, FIGURE_HEIGHT)
from ibl_pipeline.analyses import behavior as behavior_analysis

# Settings
fig_path = figpath()
bin_centers = np.arange(3, 40, 3)
bin_size = 5
seaborn_style()

# Load in data
use_subjects = query_subjects()
behav = (use_subjects * behavior_analysis.BehavioralSummaryByDate).fetch(format='frame')
behav['lab'] = behav['institution_short']
behav['lab_number'] = behav.lab.map(institution_map()[0])

# Get variability over days
mean_days = pd.DataFrame(columns=bin_centers, index=np.unique(behav['lab_number']))
std_days = pd.DataFrame(columns=bin_centers, index=np.unique(behav['lab_number']))
for i, day in enumerate(bin_centers):
    this_behav = behav[(behav['training_day'] > day - np.floor(bin_size / 2))
                       & (behav['training_day'] < day + np.floor(bin_size / 2))]
    mean_days[day] = this_behav.groupby('lab_number').mean()['performance_easy']
    std_days[day] = this_behav.groupby('lab_number').std()['performance_easy']

# Plot output

colors = group_colors()
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(FIGURE_WIDTH*0.7, FIGURE_HEIGHT))
for i, lab in enumerate(std_days.index.values):
    ax1.plot(std_days.loc[lab], color=colors[i], lw=2, label='Lab %s' % (i + 1))
    #ax1.legend(frameon=False, loc='lower center', ncol=3, bbox_to_anchor=(0.5, 1))
ax1.set(xlabel='Training days', ylabel='Variability (std)', title='Within labs')
ax1.set(xlim=[0, 40])
ax2.plot(mean_days.std(), lw=2)
ax2.set(xlabel='Training days', ylabel='Variability (std)', title='Between labs')
ax2.set(xlim=[0, 40])

sns.despine(trim=True)
plt.tight_layout()
plt.savefig(join(fig_path, 'suppfig4_variability_over_time.pdf'))
plt.savefig(join(fig_path, 'suppfig4_variability_over_time.png'), dpi=300)

### The same but for trials ###

# Settings
bin_size = 1000
bin_centers = np.arange(1000, 30001, 1000)

# Create column for cumulative trials per mouse
behav.n_trials_date = behav.n_trials_date.astype(int)
behav['cum_trials'] = (
    (behav
        .groupby(by=['subject_uuid'])
        .cumsum()
        .n_trials_date)
)

# Get variability over days
mean_trials = pd.DataFrame(columns=bin_centers, index=np.unique(behav['lab_number']))
std_trials = pd.DataFrame(columns=bin_centers, index=np.unique(behav['lab_number']))
for i, day in enumerate(bin_centers):
    this_behav = behav[(behav['cum_trials'] > day - np.floor(bin_size / 2))
                       & (behav['cum_trials'] < day + np.floor(bin_size / 2))]
    mean_trials[day] = this_behav.groupby('lab_number').mean()['performance_easy']
    std_trials[day] = this_behav.groupby('lab_number').std()['performance_easy']

# Plot output

xlim = [0, 30000]
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(FIGURE_WIDTH * 0.7, FIGURE_HEIGHT))
for i, lab in enumerate(std_trials.index.values):
    ax1.plot(std_trials.loc[lab], color=colors[i], lw=2, label='Lab %s' % (i + 1))
ax1.set(xlabel='Cumulative trials', ylabel='Variability (std)', title='Within labs')
ax1.set(xlim=xlim)
ax2.plot(mean_trials.std(), lw=2)
ax2.set(xlabel='Cumulative trials', ylabel='Variability (std)', title='Between labs')
ax2.set(xlim=xlim)

sns.despine(trim=True)
plt.tight_layout()
plt.savefig(join(fig_path, 'suppfig4_variability_over_trials.pdf'))
plt.savefig(join(fig_path, 'suppfig4_variability_over_trials.png'), dpi=300)
