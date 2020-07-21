#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 2020-07-20
@author: Anne Urai
"""
import datajoint as dj
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from paper_behavior_functions import (seaborn_style, institution_map,
                                      group_colors, figpath, EXAMPLE_MOUSE,
                                      FIGURE_WIDTH, FIGURE_HEIGHT)
from dj_tools import num_star
import os
from scipy import stats

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
combined = params_basic.merge(params_full, on=['institution_code', 'subject_nickname'])

# ========================================== #
# PRINT SUMMARY AND STATS
# ========================================== #

vars = ['6.25', '12.5', '25', '100', 'rewarded','unrewarded',  'bias']
for v in vars:
    print('basic task, %s: mean %.2f, %f : %f'%(v, params_basic[v].mean(),
                                           params_basic[v].min(),
                                           params_basic[v].max()))

    print('full task, %s: mean %.2f, %f : %f'%(v, params_full[v].mean(),
                                           params_full[v].min(),
                                           params_full[v].max()))

    # DO STATS BETWEEN THE TWO TASK TYPES
    test = stats.ttest_rel(combined[v + '_y'],
                           combined[v + '_x'],
                           axis=0, nan_policy='omit')
    print(test)

# just show the average block bias in the full task
print('full task, block_id: mean %.2f, %f: %f'%(params_full['block_id'].mean(),
                                       params_full['block_id'].min(),
                                       params_full['block_id'].max()))

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
              palette = pal, marker='o', ax=ax[0], zorder=0, edgecolors='white')
ax[0].plot(basic_summ_visual.groupby(['variable'])['value'].mean()[['6.25', '12.5', '25', '100']],
             color='black', linewidth=0, marker='_', markersize=13)
ax[0].get_legend().set_visible(False)
ax[0].set(xlabel='  ', ylabel='Weight', ylim=[0,5.5])

sns.swarmplot(data = basic_summ_bias,
              hue = 'institution_code', x = 'variable', y= 'value',
              order=['rewarded', 'unrewarded', 'bias'],
              palette = pal, marker='o', ax=ax[1], zorder=0, edgecolors='white')
ax[1].plot(basic_summ_bias.groupby(['variable'])['value'].mean()[['rewarded', 'unrewarded', 'bias']],
             color='black', linewidth=0, marker='_', markersize=13)
ax[1].get_legend().set_visible(False)
ax[1].set(xlabel='', ylabel='', ylim=[-0.5,1.2], yticks=[-0.5, 0, 0.5, 1],
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
# WEIGHTS IN THE FULL TASK
plt.close('all')
fig, ax  = plt.subplots(1, 2, figsize=(FIGURE_WIDTH/3, FIGURE_HEIGHT))
sns.swarmplot(data = full_summ_visual,
              order=['6.25', '12.5', '25', '100'],
              hue = 'institution_code', x = 'variable', y= 'value',
              palette = pal, marker='o', ax=ax[0], zorder=0, edgecolor='white')
ax[0].plot(full_summ_visual.groupby(['variable'])['value'].mean()[['6.25', '12.5', '25', '100']],
             color='black', linewidth=0, marker='_', markersize=13)
ax[0].get_legend().set_visible(False)
ax[0].set(xlabel=' ', ylabel='Weight', ylim=[0,5.5])

sns.swarmplot(data = full_summ_bias,
              hue = 'institution_code', x = 'variable', y= 'value',
              order=['rewarded', 'unrewarded', 'bias', 'block_id'],
              palette = pal, marker='o', ax=ax[1], zorder=0, edgecolor='white')
ax[1].plot(full_summ_bias.groupby(['variable'])['value'].mean()[['rewarded', 'unrewarded', 'bias', 'block_id']],
             color='black', linewidth=0, marker='_', markersize=13)
ax[1].get_legend().set_visible(False)
ax[1].set(xlabel='', ylabel='', ylim=[-0.5,1.2], yticks=[-0.5, 0, 0.5, 1])
ax[1].set_xticklabels([], ha='right', rotation=20)

sns.despine(trim=True)
plt.tight_layout(w_pad=-0.1)
fig.savefig(os.path.join(figpath, 'figure5c_full_weights.pdf'))

# ========================================== #
#%% SUPPLEMENTARY FIGURE:
# EACH PARAMETER ACROSS LABS
# ========================================== #

# add the data for all labs combined
params_basic_all = params_basic.copy()
params_basic_all['institution_code'] = 'All'
params_basic_all = params_basic.append(params_basic_all)

# add the data for all labs combined
params_full_all = params_full.copy()
params_full_all['institution_code'] = 'All'
params_full_all = params_full.append(params_full_all)

# which variables to plot?
vars = ['6.25', '12.5', '25', '100', 'unrewarded', 'rewarded', 'bias', 'block_id', 'pseudo_rsq', 'accuracy']
ylabels =['Contrast: 6.25', 'Contrast: 12.5', 'Contrast: 25', ' Contrast: 100',
          'Past choice: unrewarded', 'Past choice: rewarded', 'Bias: constant',
          'Bias: block prior', 'Pseudo-R$^2$', 'Model accuracy (5-fold c.v.)']
ylims = [[0, 6.5], [0, 6.5], [0, 6.5], [0, 6.5], [-1, 1.5], [-1, 1.5],
         [-2, 2], [-0.5, 1], [0, 1], [0.5, 1.02]]

plt.close('all')
for params, modelname in zip([[params_basic, params_basic_all],
                              [params_full, params_full_all]], ['basic', 'full']):
    for v, ylab, ylim in zip(vars, ylabels, ylims):

        if v in params[0].columns: # skip bias for the basic task

            print(modelname)
            print(v)
            f, ax = plt.subplots(1, 1, figsize=(FIGURE_WIDTH/5, FIGURE_HEIGHT))
            sns.swarmplot(y=v, x='institution_code', data=params[0], hue='institution_code',
                          palette=pal, ax=ax, marker='.')
            axbox = sns.boxplot(y=v, x='institution_code', data=params[1], color='white',
                                showfliers=False, ax=ax)
            ax.set(ylabel=ylab, xlabel='', ylim=ylim)
            # [tick.set_color(lab_colors[i]) for i, tick in enumerate(ax5.get_xticklabels()[:-1])]
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=60)
            axbox.artists[-1].set_edgecolor('black')
            for j in range(5 * (len(axbox.artists) - 1), 5 * len(axbox.artists)):
                axbox.lines[j].set_color('black')
            ax.get_legend().set_visible(False)

            # DO STATISTICS
            _, normal = stats.normaltest(params[0][v], nan_policy='omit')

            if normal < 0.05:
                test_type = 'kruskal'
                test = stats.kruskal(*[group[v].values
                                       for name, group in params[0].groupby('institution_code')],
                                     nan_policy='omit')
            else:
                test_type = 'anova'
                test = stats.f_oneway(*[group[v].values
                                        for name, group in params[0].groupby('institution_code')])

            # statistical annotation
            pvalue = test[1]
            if pvalue < 0.05:
                ax.annotate(num_star(pvalue),
                                 xy=[0.1, 0.8], xycoords='axes fraction', fontsize=5)

            sns.despine(trim=True)
            plt.tight_layout()
            plt.savefig(os.path.join(figpath, 'suppfig_model_%s_metrics_%s.pdf'%(modelname, v)))
