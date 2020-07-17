#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 12 16:49:07 2020

@author: Anne Urai
"""
import datajoint as dj
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from paper_behavior_functions import (query_sessions_around_criterion,
                                      seaborn_style, institution_map, 
                                      group_colors, figpath, EXAMPLE_MOUSE,
                                      FIGURE_WIDTH, FIGURE_HEIGHT)
from dj_tools import dj2pandas, fit_psychfunc
import os

# Load some things from paper_behavior_functions
figpath = figpath()
seaborn_style()
institution_map, col_names = institution_map()
pal = group_colors()
cmap = sns.diverging_palette(20, 220, n=3, center="dark")

# ========================================== #
#%% 1. GET GLM FITS FOR ALL MICE
# ========================================== #

print('loading model from disk...')
params_basic = pd.read_csv('./model_results/params_basic.csv')
params_full = pd.read_csv('./model_results/params_full.csv')

# ========================================== #
#%% 2. PLOT WEIGHTS ACROSS MICE AND LABS
# ========================================== #

# reshape the data and average across labs for easy plotting
basic_summ_visual = pd.melt(params_basic,
                     id_vars=['institution_code', 'subject_nickname'],
                     value_vars=['6.25', '12.5', '25', '100']).groupby(['institution_code',
                                                     'variable']).mean().reset_index()
basic_summ_bias = pd.melt(params_basic,
                     id_vars=['institution_code', 'subject_nickname'],
                     value_vars=['unrewarded', 'rewarded', 'bias']).groupby(['institution_code',
                                                     'variable']).mean().reset_index()
# WEIGHTS IN THE BASIC TASK
plt.close('all')
fig, ax = plt.subplots(1, 2, figsize=(FIGURE_WIDTH/3, FIGURE_HEIGHT))
sns.swarmplot(data = basic_summ_visual,
              hue = 'institution_code', x = 'variable', y= 'value',
              order=['6.25', '12.5', '25', '100'],
              palette = pal, marker='.', ax=ax[0], zorder=0)
ax[0].plot(basic_summ_visual.groupby(['variable'])['value'].mean()[['6.25', '12.5', '25', '100']],
             color='black', linewidth=0, marker='_', markersize=10)
ax[0].get_legend().set_visible(False)
ax[0].set(xlabel='  ', ylabel='Weight', ylim=[0,5])

sns.swarmplot(data = basic_summ_bias,
              hue = 'institution_code', x = 'variable', y= 'value',
              order=['rewarded', 'unrewarded', 'bias'],
              palette = pal, marker='.', ax=ax[1], zorder=0)
ax[1].plot(basic_summ_bias.groupby(['variable'])['value'].mean()[['rewarded', 'unrewarded', 'bias']],
             color='black', linewidth=0, marker='_', markersize=10)
ax[1].get_legend().set_visible(False)
ax[1].set(xlabel='', ylabel='', ylim=[-0.5,1], yticks=[-0.5, 0, 0.5, 1],
          xticks=[0,1,2,3], xlim=[-0.5, 3.5])
ax[1].set_xticklabels([], ha='right', rotation=20)
sns.despine(trim=True)
plt.tight_layout(w_pad=-0.1)
fig.savefig(os.path.join(figpath, 'figure5c_basic_weights.pdf'))

# ========================= #
# SAME BUT FOR FULL TASK
# ========================= #

# reshape the data and average across labs for easy plotting
full_summ_visual = pd.melt(params_full,
                     id_vars=['institution_code', 'subject_nickname'],
                     value_vars=['6.25', '12.5', '25', '100']).groupby(['institution_code',
                                                     'variable']).mean().reset_index()
full_summ_bias = pd.melt(params_full,
                     id_vars=['institution_code', 'subject_nickname'],
                     value_vars=['unrewarded', 'rewarded',
                                 'bias', 'block_id']).groupby(['institution_code',
                                                     'variable']).mean().reset_index()
# WEIGHTS IN THE BASIC TASK
plt.close('all')
fig, ax  = plt.subplots(1, 2, figsize=(FIGURE_WIDTH/3, FIGURE_HEIGHT))
sns.swarmplot(data = full_summ_visual,
              order=['6.25', '12.5', '25', '100'],
              hue = 'institution_code', x = 'variable', y= 'value',
              palette = pal, marker='.', ax=ax[0], zorder=0)
ax[0].plot(full_summ_visual.groupby(['variable'])['value'].mean()[['6.25', '12.5', '25', '100']],
             color='black', linewidth=0, marker='_', markersize=10)
ax[0].get_legend().set_visible(False)
ax[0].set(xlabel=' ', ylabel='Weight', ylim=[0,5])

sns.swarmplot(data = full_summ_bias,
              hue = 'institution_code', x = 'variable', y= 'value',
              order=['rewarded', 'unrewarded', 'bias', 'block_id'],
              palette = pal, marker='.', ax=ax[1], zorder=0)
ax[1].plot(full_summ_bias.groupby(['variable'])['value'].mean()[['rewarded', 'unrewarded', 'bias', 'block_id']],
             color='black', linewidth=0, marker='_', markersize=10)
ax[1].get_legend().set_visible(False)
ax[1].set(xlabel='', ylabel='', ylim=[-0.5,1], yticks=[-0.5, 0, 0.5, 1])
ax[1].set_xticklabels([], ha='right', rotation=20)

sns.despine(trim=True)
plt.tight_layout(w_pad=-0.1)
fig.savefig(os.path.join(figpath, 'figure5c_full_weights.pdf'))

# ========================================== #
#%% SUPPLEMENTARY FIGURE:
# EACH PARAMETER ACROSS LABS
# ========================================== #

