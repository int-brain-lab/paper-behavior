#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 21 10:30:25 2018

Quantify the bias shift within and between labs of mouse behavior.
This script doesn't perform any analysis but plots summary statistics over labs.

@author: Guido Meijer
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from os.path import join, expanduser
import seaborn as sns
from paper_behavior_functions import (query_sessions_around_criterion, seaborn_style,
                                      institution_map, group_colors)
from ibl_pipeline import subject, reference
from ibl_pipeline.analyses import behavior as behavioral_analyses

# Settings
fig_path = join(expanduser('~'), 'Figures', 'Behavior')

# Query sessions
sessions = query_sessions_around_criterion(criterion='ephys', days_from_criterion=[2, 0])[0]
sessions = sessions * subject.Subject * subject.SubjectLab * reference.Lab

# Get bias per block
biased_blocks = (sessions * behavioral_analyses.PsychResultsBlock).fetch(format='frame')
biased_blocks = biased_blocks.reset_index()

# Select 20 and 80 blocks
biased_blocks = (biased_blocks[(biased_blocks['prob_left_block'] == 20)
                 | (biased_blocks['prob_left_block'] == 80)])

# Create dataframe
bias = pd.DataFrame(columns=['mouse', 'lab', 'bias_20', 'bias_80', 'bias_shift'])

# Get bias shift per session
for i, nickname in enumerate(np.unique(biased_blocks['subject_nickname'])):

    # Calculte bias shift for this subject
    shift = (biased_blocks.loc[((biased_blocks['prob_left_block'] == 80)
                                & (biased_blocks['subject_nickname'] == nickname)),
                               'bias'].values
             - biased_blocks.loc[((biased_blocks['prob_left_block'] == 20)
                                  & (biased_blocks['subject_nickname'] == nickname)),
                                 'bias'].values)

    # Add to dataframe
    bias.loc[i, 'mouse'] = nickname
    bias.loc[i, 'lab'] = biased_blocks.loc[biased_blocks['subject_nickname'] == nickname,
                                           'institution_short'].values[0]
    bias.loc[i, 'bias_shift'] = np.mean(shift)
    bias.loc[i, 'bias_20'] = np.mean(biased_blocks.loc[
            ((biased_blocks['prob_left_block'] == 20)
             & (biased_blocks['subject_nickname'] == nickname)), 'bias'])
    bias.loc[i, 'bias_80'] = np.mean(biased_blocks.loc[
            ((biased_blocks['prob_left_block'] == 80)
             & (biased_blocks['subject_nickname'] == nickname)), 'bias'])

# Change lab name into lab number
bias['lab_number'] = bias.lab.map(institution_map()[0])
bias = bias.sort_values('lab_number')

# Convert to float
bias['bias_shift'] = bias['bias_shift'].astype(float)
bias['bias_20'] = bias['bias_20'].astype(float)
bias['bias_80'] = bias['bias_80'].astype(float)

# Add all mice to dataframe seperately for plotting
bias_all = bias.copy()
bias_all['lab_number'] = 'All'
bias_all = bias.append(bias_all)

# Set color palette
use_palette = [[0.6, 0.6, 0.6]] * len(np.unique(bias['lab']))
use_palette = use_palette + [[1, 1, 0.2]]
sns.set_palette(use_palette)
lab_colors = group_colors()

# Plot behavioral metrics per lab
f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(10, 4))
sns.set_palette(use_palette)

sns.boxplot(y='bias_20', x='lab_number', data=bias_all, ax=ax1)
ax1.set(ylabel='Bias', ylim=[-25, 25], xlabel='', title='Probability left: 20%')
[tick.set_color(lab_colors[i]) for i, tick in enumerate(ax1.get_xticklabels()[:-1])]
plt.setp(ax1.xaxis.get_majorticklabels(), rotation=40)

sns.boxplot(y='bias_80', x='lab_number', data=bias_all, ax=ax2)
ax2.set(ylabel='Bias', ylim=[-25, 25], xlabel='', title='Probability left: 80%')
[tick.set_color(lab_colors[i]) for i, tick in enumerate(ax2.get_xticklabels()[:-1])]
plt.setp(ax2.xaxis.get_majorticklabels(), rotation=40)

sns.boxplot(y='bias_shift', x='lab_number', data=bias_all, ax=ax3)
ax3.set(ylabel='Bias shift', ylim=[0, 25], xlabel='')
[tick.set_color(lab_colors[i]) for i, tick in enumerate(ax3.get_xticklabels()[:-1])]
plt.setp(ax3.xaxis.get_majorticklabels(), rotation=40)

plt.tight_layout(pad=2)
seaborn_style()
plt.savefig(join(fig_path, 'figure4_bias_per_lab.pdf'), dpi=300)
plt.savefig(join(fig_path, 'figure4_bias_per_lab.png'), dpi=300)
