"""
Learning curves for all labs
​
@author: Anne Urai, Alejandro Pan-Vazquez
31 March 2021
"""
import os
import pandas as pd
import numpy as np
from scipy.signal import medfilt
import seaborn as sns
import matplotlib.pyplot as plt
from paper_behavior_functions import (query_subjects, figpath, datapath, group_colors,
                                      institution_map, seaborn_style, EXAMPLE_MOUSE,
                                      FIGURE_HEIGHT, FIGURE_WIDTH, QUERY)
from ibl_pipeline.analyses import behavior as behavioral_analyses
# INITIALIZE A FEW THINGS
seaborn_style()
figpath = figpath()
pal = group_colors()
institution_map, col_names = institution_map()
col_names = col_names[:-1]
# %% ============================== #
# GET DATA FROM TRAINED ANIMALS
# ================================= #
if QUERY is True:
    use_subjects = query_subjects()
    b = (behavioral_analyses.BehavioralSummaryByDate * use_subjects * behavioral_analyses.BehavioralSummaryByDate.PsychResults)
    behav = b.fetch(order_by='institution_short, subject_nickname, training_day',
                    format='frame').reset_index()
    behav['institution_code'] = behav.institution_short.map(institution_map)
else:
    behav = pd.read_csv(os.path.join(datapath(), 'Fig2ab.csv'))
# exclude sessions with fewer than 100 trials
behav = behav[behav['n_trials_date'] > 100]
# exclude sessions with less than 3 types of contrast
behav.loc[behav['signed_contrasts'].str.len()<6,'threshold'] = np.nan
behav.loc[behav['signed_contrasts'].str.len()<6,'bias'] = np.nan
# convolve performance over 3 days
for i, nickname in enumerate(behav['subject_nickname'].unique()):
    # 1.Performance
    perf = behav.loc[behav['subject_nickname'] == nickname, 'performance_easy'].values
    perf_conv = np.convolve(perf, np.ones((3,))/3, mode='valid')
    # perf_conv = np.append(perf_conv, [np.nan, np.nan])
    perf_conv = medfilt(perf, kernel_size=3)
    behav.loc[behav['subject_nickname'] == nickname, 'performance_easy'] = perf_conv
    # 2.Threshold
    thre = behav.loc[behav['subject_nickname'] == nickname,
                     'threshold'].values
    thre_conv = np.convolve(thre, np.ones((3,))/3, mode='valid')
    # perf_conv = np.append(perf_conv, [np.nan, np.nan])
    thre_conv = medfilt(thre, kernel_size=3)
    behav.loc[behav['subject_nickname'] == nickname,
              'threshold'] = thre_conv
    # 3.Bias
    bias = behav.loc[behav['subject_nickname'] == nickname, 'bias'].values
    bias_conv = np.convolve(bias, np.ones((3,))/3, mode='valid')
    # perf_conv = np.append(perf_conv, [np.nan, np.nan])
    bias_conv = medfilt(bias, kernel_size=3)
    behav.loc[behav['subject_nickname'] == nickname, 'conv_bias'] = bias_conv

# exclude sessions with less than 3 types of contrast
behav.loc[behav['signed_contrasts'].str.len()<6,'threshold'] = np.nan
behav.loc[behav['signed_contrasts'].str.len()<6,'bias_conv'] = np.nan

# how many mice are there for each lab?
N = behav.groupby(['institution_code'])['subject_nickname'].nunique().to_dict()
behav['n_mice'] = behav.institution_code.map(N)
behav['institution_name'] = behav.institution_code + '\n ' + behav.n_mice.apply(str) + ' mice'
# make sure each mouse starts at 0
for index, group in behav.groupby(['lab_name', 'subject_nickname']):
    behav.loc[group.index, 'training_day'] = group['training_day'] - group['training_day'].min()

# create another column only after the mouse is trained
behav2 = pd.DataFrame([])
for index, group in behav.groupby(['institution_code', 'subject_nickname']):
    group['performance_easy_trained'] = group.performance_easy
    group.loc[pd.to_datetime(group['session_date']) < pd.to_datetime(group['date_trained']),
              'performance_easy_trained'] = np.nan
    group['threshold_easy_trained'] = group.threshold
    group.loc[pd.to_datetime(group['session_date']) < pd.to_datetime(group['date_trained']),
              'threshold_easy_trained'] = np.nan
    group['bias_trained'] = group.conv_bias
    group.loc[pd.to_datetime(group['session_date']) < pd.to_datetime(group['date_trained']),
              'bias_trained'] = np.nan
    # add this
    behav2 = behav2.append(group)
behav = behav2
behav['performance_easy'] = behav.performance_easy * 100
behav['performance_easy_trained'] = behav.performance_easy_trained * 100

behav = behav.loc[behav['prob_left']==0.5]

# %% ============================== #
# LEARNING CURVES
# ================================= #
# Performance: plot one curve for each animal, one panel per lab
fig = sns.FacetGrid(behav,
                    col="institution_code", col_wrap=7, col_order=col_names,
                    sharex=True, sharey=True, hue="subject_uuid", xlim=[-1, 40],
                    height=FIGURE_HEIGHT, aspect=(FIGURE_WIDTH/7)/FIGURE_HEIGHT)
fig.map(sns.lineplot, "training_day",
        "performance_easy", color='grey', alpha=0.3)
fig.map(sns.lineplot, "training_day",
        "performance_easy_trained", color='black', alpha=0.3)
fig.set_titles("{col_name}")
fig.set(xticks=[0, 20, 40])

# overlay the example mouse
sns.lineplot(ax=fig.axes[0], x='training_day', y='performance_easy', color='black',
             data=behav[behav['subject_nickname'].str.contains(EXAMPLE_MOUSE)], legend=False)

for axidx, ax in enumerate(fig.axes.flat):
    # add the lab mean to each panel
    sns.lineplot(data=behav.loc[behav.institution_name == behav.institution_name.unique()[axidx], :],
                 x='training_day', y='performance_easy',
                 color=pal[axidx], ci=None, ax=ax, legend=False, linewidth=2)
    ax.set_title(behav.institution_name.unique()[
                 axidx], color=pal[axidx], fontweight='bold')

fig.set_axis_labels('  ', 'Performance (%) on easy trials')
fig.despine(trim=True)
plt.tight_layout()
fig.savefig(os.path.join(figpath, "figure2a_learningcurves.pdf"))
fig.savefig(os.path.join(figpath, "figure2a_learningcurves.png"), dpi=300)

# Threshold: plot one curve for each animal, one panel per lab
fig = sns.FacetGrid(behav,
                    col="institution_code", col_wrap=7, col_order=col_names,
                    sharex=True, sharey=True, hue="subject_uuid", xlim=[-1, 40],
                    ylim=[0, 40],
                    height=FIGURE_HEIGHT, aspect=(FIGURE_WIDTH/7)/FIGURE_HEIGHT)
fig.map(sns.lineplot, "training_day",
        "threshold", color='grey', alpha=0.3)
fig.map(sns.lineplot, "training_day",
        "threshold_easy_trained", color='black', alpha=0.3)
fig.set_titles("{col_name}")
fig.set(xticks=[0, 20, 40])
# overlay the example mouse
sns.lineplot(ax=fig.axes[0], x='training_day', y='threshold',
             color='black',
             data=behav[behav['subject_nickname'].str.contains(EXAMPLE_MOUSE)],
             legend=False)
behav_sum_threshold=pd.DataFrame()
for axidx, ax in enumerate(fig.axes.flat):
    # First day with at least 3 threshold values
    start_day = np.where(behav.loc[behav.institution_name == behav.institution_name.unique()[axidx], :]\
        .groupby(['training_day']).count()['threshold']>2)[0][0]
    # add the lab mean to each panel
    sns.lineplot(data=behav.loc[(behav.institution_name == behav.institution_name.unique()[axidx]) &
                      (behav['training_day']>=start_day), :],
                 x='training_day', y='threshold',
                 color=pal[axidx], ci=None, ax=ax, legend=False, linewidth=2)
    ax.set_title(behav.institution_name.unique()[
                     axidx], color=pal[axidx], fontweight='bold')
    behav_sum_threshold = behav_sum_threshold.append(behav.loc[(behav.institution_name == behav.institution_name.unique()[axidx]) &
                      (behav['training_day']>=start_day), :])

fig.set_axis_labels('  ', 'Threshold')
fig.despine(trim=True)
plt.tight_layout()
fig.savefig(os.path.join(figpath, "figure2b_threshold.pdf"))
fig.savefig(os.path.join(figpath, "figure2b_threshold.png"), dpi=300)

# Bias: plot one curve for each animal, one panel per lab
fig = sns.FacetGrid(behav,
                    col="institution_code", col_wrap=7, col_order=col_names,
                    sharex=True, sharey=True, hue="subject_uuid", xlim=[-1, 40],
                    ylim=[-30, 30],
                    height=FIGURE_HEIGHT, aspect=(FIGURE_WIDTH/7)/FIGURE_HEIGHT)
fig.map(sns.lineplot, "training_day",
        "conv_bias", color='grey', alpha=0.3)
fig.map(sns.lineplot, "training_day",
        "bias_trained", color='black', alpha=0.3)
fig.set_titles("{col_name}")
fig.set(xticks=[0, 20, 40])
# overlay the example mouse
sns.lineplot(ax=fig.axes[0], x='training_day', y='bias', color='black',
             data=behav[behav['subject_nickname'].str.contains(EXAMPLE_MOUSE)], legend=False)
behav_sum_bias=pd.DataFrame()
for axidx, ax in enumerate(fig.axes.flat):
    # First day with at least 3 bias values
    start_day = np.where(behav.loc[behav.institution_name == behav.institution_name.unique()[axidx], :]\
        .groupby(['training_day']).count()['conv_bias']>2)[0][0]
    # add the lab mean to each panel
    sns.lineplot(data=behav.loc[(behav.institution_name == behav.institution_name.unique()[axidx]) &
                      (behav['training_day']>=start_day), :],
                 x='training_day', y='conv_bias',
                 color=pal[axidx], ci=None, ax=ax, legend=False, linewidth=2)
    ax.set_title(behav.institution_name.unique()[
                     axidx], color=pal[axidx], fontweight='bold')
    behav_sum_bias=behav_sum_bias.append(behav.loc[(behav.institution_name == behav.institution_name.unique()[axidx]) &
                      (behav['training_day']>=start_day), :])
fig.set_axis_labels('Training day', 'Bias (%)')
fig.despine(trim=True)
plt.tight_layout()
fig.savefig(os.path.join(figpath, "figure2c_bias.pdf"))
fig.savefig(os.path.join(figpath, "figure2c_bias.png"), dpi=300)
# Plot all labs
# 1. Performance
fig, ax1 = plt.subplots(1, 1, figsize=(FIGURE_WIDTH/5, FIGURE_HEIGHT))
sns.lineplot(x='training_day', y='performance_easy', hue='institution_code', palette=pal,
             ax=ax1, legend=False, data=behav, ci=None)
sns.lineplot(x='training_day', y='performance_easy', color='k',
             ax=ax1, legend=False, data=behav, ci=None)
ax1.set_title('All labs: %d mice'%behav['subject_nickname'].nunique())
ax1.set(xlabel='Training day',
        ylabel='Performance (%)\non easy trials', xlim=[-1, 60], ylim=[15,100])
ax1.set(xticks=[0, 20, 40, 60])
ax1.set_title('All labs: %d mice'%behav['subject_nickname'].nunique())
sns.despine(trim=True)
plt.tight_layout()
fig.savefig(os.path.join(figpath, "figure2d_learningcurves_all_labs.pdf"))
fig.savefig(os.path.join(
    figpath, "figure2d_learningcurves_all_labs.png"), dpi=300)
# 2. Threshold
# day from which we have data from all labs
# new dataframe including summary per lab from 3 mice or more per day
fig, ax1 = plt.subplots(1, 1, figsize=(FIGURE_WIDTH/5, FIGURE_HEIGHT))
sns.lineplot(x='training_day', y='threshold', hue='institution_code', palette=pal,
             ax=ax1, legend=False, data=behav_sum_threshold, ci=None)
sns.lineplot(x='training_day', y='threshold', color='k',
             ax=ax1, legend=False, data=behav_sum_threshold, ci=None)
ax1.set(xlabel='Training day',
        ylabel='Threshold', xlim=[-1, 60], ylim=[0,40])
ax1.set(xticks=[0, 20, 40, 60])
ax1.set_title('All labs: %d mice'%behav['subject_nickname'].nunique())
sns.despine(trim=True)
plt.tight_layout()
fig.savefig(os.path.join(figpath, "2e_threshold_all_labs.pdf"))
fig.savefig(os.path.join(
    figpath, "2e_threshold_all_labs.png"), dpi=300)
# 3. Bias
fig, ax1 = plt.subplots(1, 1, figsize=(FIGURE_WIDTH/5, FIGURE_HEIGHT))
sns.lineplot(x='training_day', y='conv_bias', hue='institution_code', palette=pal,
             ax=ax1, legend=False, data=behav_sum_bias, ci=None)
sns.lineplot(x='training_day', y='conv_bias', color='k',
             ax=ax1, legend=False, data=behav_sum_bias, ci=None)
ax1.set(xlabel='Training day',
        ylabel='Bias (%)', xlim=[-1, 60], ylim=[-30,30])
ax1.set(xticks=[0, 20, 40, 60])
ax1.set_title('All labs: %d mice'%behav['subject_nickname'].nunique())
sns.despine(trim=True)
plt.tight_layout()
fig.savefig(os.path.join(figpath, "figure2f_learningcurves_all_labs_bias.pdf"))
fig.savefig(os.path.join(
    figpath, "2f_learningcurves_all_labs_bias.png"), dpi=300)
# ================================= #
# print some stats
# ================================= #
behav_summary_std = behav.groupby(['training_day'])[
    'performance_easy'].std().reset_index()
behav_summary = behav.groupby(['training_day'])[
    'performance_easy'].mean().reset_index()
print('number of days to reach 80% accuracy on easy trials: ')
print(behav_summary.loc[behav_summary.performance_easy >
                        80, 'training_day'].min())
