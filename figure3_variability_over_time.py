#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 21 10:30:25 2018

Quantify the variability of behavioral metrics within and between labs of mouse behavior.
This script doesn't perform any analysis but plots summary statistics over labs.

@author: Guido Meijer
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from os.path import join
import seaborn as sns
from paper_behavior_functions import (query_sessions_around_criterion, seaborn_style,
                                      institution_map, group_colors, figpath, query_subjects)
from dj_tools import dj2pandas, fit_psychfunc
from ibl_pipeline import behavior, subject, reference
import scikit_posthocs as sp
from ibl_pipeline.analyses import behavior as behavior_analysis

# Settings
fig_path = figpath()
bin_centers = np.arange(3, 40, 3)
bin_size = 5

# Load in data
use_subjects = query_subjects()
behav = (use_subjects * behavior_analysis.BehavioralSummaryByDate).fetch(format='frame')
behav['lab'] = behav['institution_short']
behav['lab_number'] = behav.lab.map(institution_map()[0])

# Get variability over days
mean_days = pd.DataFrame(columns=bin_centers, index=np.unique(behav['lab_number']))
std_days = pd.DataFrame(columns=bin_centers, index=np.unique(behav['lab_number']))
for i, day in enumerate(bin_centers):
    this_behav = behav[(behav['training_day'] > day-np.floor(bin_size/2))
                       & (behav['training_day'] < day+np.floor(bin_size/2))]
    mean_days[day] = this_behav.groupby('lab_number').mean()['performance_easy']
    std_days[day] = this_behav.groupby('lab_number').std()['performance_easy']

# Plot output
colors = group_colors()
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))
for i, lab in enumerate(std_days.index.values):
    ax1.plot(std_days.loc[lab], color=colors[i], lw=2)
ax1.set(xlabel='Training days', ylabel='Variability (std)', title='Within labs')

ax2.plot(mean_days.std(), lw=2)
ax2.set(xlabel='Training days', ylabel='Variability (std)', title='Between labs')


plt.tight_layout(pad=2)
plt.savefig(join(fig_path, 'variability_over_time.pdf'), dpi=300)
plt.savefig(join(fig_path, 'variability_over_time.png'), dpi=300)
