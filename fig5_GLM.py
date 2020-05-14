#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 12 16:49:07 2020

@author: alex
"""
import datajoint as dj
dj.config['database.host'] = 'datajoint.internationalbrainlab.org'
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from os.path import join
import seaborn as sns
from paper_behavior_functions import (query_sessions_around_criterion, seaborn_style,
                                      institution_map, group_colors, figpath)
from dj_tools import dj2pandas, fit_psychfunc
from ibl_pipeline import behavior, subject, reference
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score, KFold
import os


##############################################################################
#*******************************Biased Task**********************************#
##############################################################################

# Query sessions biased data 
use_sessions, use_days = query_sessions_around_criterion(criterion='biased',
                                                         days_from_criterion=[
                                                             2, 3],
                                                         as_dataframe=False)
institution_map, col_names = institution_map()

# restrict by list of dicts with uuids for these sessions
b = (use_sessions * subject.Subject * subject.SubjectLab * reference.Lab
     * behavior.TrialSet.Trial)

# reduce the size of the fetch
b2 = b.proj('institution_short', 'subject_nickname', 'task_protocol',
            'trial_stim_contrast_left', 'trial_stim_contrast_right', 
            'trial_response_choice', 'task_protocol', 'trial_stim_prob_left', 
            'trial_feedback_type')
bdat = b2.fetch(order_by='institution_short, subject_nickname, session_start_time, trial_id',
                format='frame').reset_index()
behav = dj2pandas(bdat)

behav['institution_code'] = behav.institution_short.map(institution_map)

# Variable to store model parameters
behav['rchoice'] = np.nan
behav['uchoice'] = np.nan
behav['6'] = np.nan
behav['12'] = np.nan
behav['25'] = np.nan
behav['100'] = np.nan
behav['block'] = np.nan
behav['intercept'] = np.nan
behav['simulation_prob'] = np.nan

# Drop trials with weird contrasts

behav.drop(behav['probabilityLeft'][~behav['probabilityLeft'].isin([50,20,80])].index,
        inplace=True)
behav.drop(behav['probabilityLeft'][~behav['signed_contrast'].isin([100,25,12,6,0,-6,-12,-25,-100])].index,
        inplace=True)

behav = behav.reset_index()
# Storage 
for i, nickname in enumerate(np.unique(behav['subject_nickname'])):
    if np.mod(i+1, 10) == 0:
        print('Loading data of subject %d of %d' % (i+1, len(
                np.unique(behav['subject_nickname']))))

    # Get the trials of the sessions around criterion
    trials = behav.loc[behav['subject_nickname'] == nickname].copy()
    
    ## GLM
    
    #make separate datafrme 
    data = trials[['index', 'trial_feedback_type',
                   'signed_contrast', 'choice',
                       'probabilityLeft']].copy()
    
    
    #drop trials with odd probabilities of left
    data.drop(
        data['probabilityLeft'][~data['probabilityLeft'].isin([50,20,80])].index,
        inplace=True)
    
    
    # Rewardeded choices: 
    data.loc[(data['choice'] == 0) &
             (data['trial_feedback_type'].isnull()), 'rchoice']  = 0 # NoGo trials
    data.loc[(data['choice'] == -1) &
             (data['trial_feedback_type'] == -1), 'rchoice']  = 0
    data.loc[(data['choice'] == -1) &
             (data['trial_feedback_type'] == 1), 'rchoice']  = -1
    data.loc[(data['choice'] == 1) &
             (data['trial_feedback_type'] == 1), 'rchoice']  = 1
    data.loc[(data['choice'] == 0) &
             (data['trial_feedback_type'].isnull()) , 'rchoice']  = 0 # NoGo trials
    data.loc[(data['choice'] == 1) &
             (data['trial_feedback_type'] == -1), 'rchoice']  = 0
    
    # Unrewarded choices:
    data.loc[(data['choice'] == 0) &
             (data['trial_feedback_type'].isnull()), 'uchoice']  = 0 # NoGo trials
    data.loc[(data['choice'] == -1) &
             (data['trial_feedback_type'] == -1), 'uchoice']  = -1 
    data.loc[(data['choice'] == -1) &
             (data['trial_feedback_type'] == 1), 'uchoice']  = 0 
    data.loc[(data['choice'] == 1) &
             (data['trial_feedback_type'] == 1), 'uchoice']  = 0 
    data.loc[(data['choice'] == 0) & 
             (data['trial_feedback_type'].isnull()) , 'uchoice']  = 0 # NoGo trials
    data.loc[(data['choice'] == 1) &
             (data['trial_feedback_type'] == -1) , 'uchoice']  = 1
        
    # Shift rewarded and unrewarded predictors by one
    data.loc[:, ['rchoice', 'uchoice']] = \
        data[['rchoice', 'uchoice']].shift(periods=1, fill_value=0).to_numpy()
    
    # Drop any nan trials
    data.dropna(inplace=True)
    
    # Make sensory predictors (no 0 predictor)
    contrasts = [ 25, 100,  12,   6]
    for i in contrasts:
        data.loc[(data['signed_contrast'].abs() == i), i] = \
            np.sign(data.loc[(data['signed_contrast'].abs() == i),
                             'signed_contrast'].to_numpy())
        
        data[i].fillna(0,  inplace=True)
    
    # Make block identity (across predictors right is positive, hence logic below)
    data.loc[(data['probabilityLeft'] == 50), 'block'] = 0
    data.loc[(data['probabilityLeft'] == 20), 'block'] = 1
    data.loc[(data['probabilityLeft'] == 80), 'block'] = -1
    
    # Make choice in between 0 and 1 -> 1 for right and 0 for left
    data.loc[data['choice'] == -1, 'choice'] = 0
    
    # Store index
    index = data['index'].copy()
    
    # Create predictor matrix
    endog = data['choice'].copy()
    exog = data.copy()
    exog.drop(columns=['trial_feedback_type', 
                   'signed_contrast', 'choice', 
                       'probabilityLeft'], inplace=True)
    exog = sm.add_constant(exog)

    X_train = exog.iloc[:int(len(exog)*0.5),:].copy()
    X_test = exog.iloc[int(len(endog)*0.5):,:].copy()
    y_train = endog.iloc[:int(len(endog)*0.5)].copy()
    y_test = endog.iloc[int(len(endog)*0.5):].copy()
    
    # Store index
    index = X_test['index'].to_numpy()
    X_train.drop(columns=['index'], inplace=True)
    X_test.drop(columns=['index'], inplace=True)
    
    # Fit model
    
    logit_model = sm.Logit(y_train, X_train)
    result = logit_model.fit()
    # print(result.summary2())
    
    # Store model weights
    behav.loc[behav['subject_nickname'] == nickname, 'intercept'] = result.params['const'].copy()
    behav.loc[behav['subject_nickname'] == nickname, 'rchoice'] = result.params['rchoice'].copy()
    behav.loc[behav['subject_nickname'] == nickname, 'uchoice'] = result.params['uchoice'].copy()
    behav.loc[behav['subject_nickname'] == nickname, '25'] = result.params[25].copy()
    behav.loc[behav['subject_nickname'] == nickname, '6'] = result.params[6].copy()
    behav.loc[behav['subject_nickname'] == nickname, '100'] = result.params[100].copy()
    behav.loc[behav['subject_nickname'] == nickname, '12'] = result.params[12].copy()
    behav.loc[behav['subject_nickname'] == nickname, 'block'] = result.params['block'].copy()
    
    # Simulate on test data with fix seed
    prob = result.predict(X_test).to_numpy()
    
    # Propagate to storing dataframe
    behav.loc[behav['index'].isin(index), 'simulation_prob'] = prob
    
##############################################################################
#*****************************Unbiased Task**********************************#
##############################################################################    


# Query sessions biased data 
use_sessions, use_days = query_sessions_around_criterion(criterion='trained',
                                                         days_from_criterion=[2, 0],
                                                         as_dataframe=False)


from paper_behavior_functions import (query_sessions_around_criterion, seaborn_style,
                                      institution_map, group_colors, figpath)


institution_map, col_names = institution_map()

# restrict by list of dicts with uuids for these sessions
t = (use_sessions * subject.Subject * subject.SubjectLab * reference.Lab
     * behavior.TrialSet.Trial)

# reduce the size of the fetch
t2 = t.proj('institution_short', 'subject_nickname', 'task_protocol',
            'trial_stim_contrast_left', 'trial_stim_contrast_right', 
            'trial_response_choice', 'task_protocol', 'trial_stim_prob_left', 
            'trial_feedback_type')
tdat = t2.fetch(order_by='institution_short, subject_nickname, session_start_time, trial_id',
                format='frame').reset_index()
tbehav = dj2pandas(tdat)

tbehav['institution_code'] = tbehav.institution_short.map(institution_map)

# Variable to store model parameters
tbehav['rchoice'] = np.nan
tbehav['uchoice'] = np.nan
tbehav['6'] = np.nan
tbehav['12'] = np.nan
tbehav['25'] = np.nan
tbehav['100'] = np.nan
tbehav['block'] = np.nan
tbehav['intercept'] = np.nan
tbehav['simulation_prob'] = np.nan

# Drop weird contrats
# Drop trials with weird contrasts

tbehav.drop(tbehav['probabilityLeft'][~tbehav['probabilityLeft'].isin([50])].index,
        inplace=True)
tbehav.drop(tbehav['probabilityLeft'][~tbehav['signed_contrast'].isin([100,25,12,6,0,-6,-12,-25,-100])].index,
        inplace=True)

tbehav = tbehav.reset_index()
# Storage 
for i, nickname in enumerate(np.unique(tbehav['subject_nickname'])):
    if np.mod(i+1, 10) == 0:
        print('Loading data of subject %d of %d' % (i+1, len(
                np.unique(tbehav['subject_nickname']))))

    # Get the trials of the sessions around criterion
    trials = tbehav.loc[tbehav['subject_nickname'] == nickname].copy()
    
    ## GLM
    
    #make separate datafrme 
    data = trials[['index', 'trial_feedback_type',
                   'signed_contrast', 'choice',
                       'probabilityLeft']].copy()
    
    
    #drop trials with odd probabilities of left
    data.drop(
        data['probabilityLeft'][~data['probabilityLeft'].isin([50])].index,
        inplace=True)
    
    
    # Rewardeded choices: 
    data.loc[(data['choice'] == 0) &
             (data['trial_feedback_type'].isnull()), 'rchoice']  = 0 # NoGo trials
    data.loc[(data['choice'] == -1) &
             (data['trial_feedback_type'] == -1), 'rchoice']  = 0
    data.loc[(data['choice'] == -1) &
             (data['trial_feedback_type'] == 1), 'rchoice']  = -1
    data.loc[(data['choice'] == 1) &
             (data['trial_feedback_type'] == 1), 'rchoice']  = 1
    data.loc[(data['choice'] == 0) &
             (data['trial_feedback_type'].isnull()) , 'rchoice']  = 0 # NoGo trials
    data.loc[(data['choice'] == 1) &
             (data['trial_feedback_type'] == -1), 'rchoice']  = 0
    
    # Unrewarded choices:
    data.loc[(data['choice'] == 0) &
             (data['trial_feedback_type'].isnull()), 'uchoice']  = 0 # NoGo trials
    data.loc[(data['choice'] == -1) &
             (data['trial_feedback_type'] == -1), 'uchoice']  = -1 
    data.loc[(data['choice'] == -1) &
             (data['trial_feedback_type'] == 1), 'uchoice']  = 0 
    data.loc[(data['choice'] == 1) &
             (data['trial_feedback_type'] == 1), 'uchoice']  = 0 
    data.loc[(data['choice'] == 0) & 
             (data['trial_feedback_type'].isnull()) , 'uchoice']  = 0 # NoGo trials
    data.loc[(data['choice'] == 1) &
             (data['trial_feedback_type'] == -1) , 'uchoice']  = 1
        
    # Shift rewarded and unrewarded predictors by one
    data.loc[:, ['rchoice', 'uchoice']] = \
        data[['rchoice', 'uchoice']].shift(periods=1, fill_value=0).to_numpy()
    
    # Drop any nan trials
    data.dropna(inplace=True)
    
    # Make sensory predictors (no 0 predictor)
    contrasts = [ 25, 100,  12,   6]
    for i in contrasts:
        data.loc[(data['signed_contrast'].abs() == i), i] = \
            np.sign(data.loc[(data['signed_contrast'].abs() == i),
                             'signed_contrast'].to_numpy())
        
        data[i].fillna(0,  inplace=True)
    
    
    # Make choice in between 0 and 1 -> 1 for right and 0 for left
    data.loc[data['choice'] == -1, 'choice'] = 0
    
    # Store index
    index = data['index'].copy()
    
    # Create predictor matrix
    endog = data['choice'].copy()
    exog = data.copy()
    exog.drop(columns=['trial_feedback_type', 
                   'signed_contrast', 'choice', 
                       'probabilityLeft'], inplace=True)
    exog = sm.add_constant(exog)

    # Can't cross-validate with half, sometimes 0.5 of the data might be missing contrasts
    X_train = exog.iloc[:int(len(exog)*0.75),:].copy()
    X_test = exog.iloc[int(len(endog)*0.75):,:].copy()
    y_train = endog.iloc[:int(len(endog)*0.75)].copy()
    y_test = endog.iloc[int(len(endog)*0.75):].copy()
    
    # Store index
    index = X_test['index'].to_numpy()
    X_train.drop(columns=['index'], inplace=True)
    X_test.drop(columns=['index'], inplace=True)
    
    # Fit model
    logit_model = sm.Logit(y_train, X_train)
    result = logit_model.fit()
    
    # print(result.summary2())
    
    # Store model weights
    tbehav.loc[tbehav['subject_nickname'] == nickname, 'intercept'] = result.params['const'].copy()
    tbehav.loc[tbehav['subject_nickname'] == nickname, 'rchoice'] = result.params['rchoice'].copy()
    tbehav.loc[tbehav['subject_nickname'] == nickname, 'uchoice'] = result.params['uchoice'].copy()
    tbehav.loc[tbehav['subject_nickname'] == nickname, '25'] = result.params[25].copy()
    tbehav.loc[tbehav['subject_nickname'] == nickname, '6'] = result.params[6].copy()
    tbehav.loc[tbehav['subject_nickname'] == nickname, '100'] = result.params[100].copy()
    tbehav.loc[tbehav['subject_nickname'] == nickname, '12'] = result.params[12].copy()
    
    # Simulate on test data with fix seed
    prob = result.predict(X_test).to_numpy()
    
    # Propagate to storing dataframe
    tbehav.loc[tbehav['index'].isin(index), 'simulation_prob'] = prob
    
##############################################################################
#******************************* Plotting ***********************************#
##############################################################################

pal = group_colors()

# Plot curve of predictors
summary_curves = pd.DataFrame()
feature_list =['100', '25', '12', '6', 'rchoice', 'uchoice', 'block', 'intercept']
cat = ['institution', 'weight', 'parameter']
for i in behav['institution_code'].unique():
    behav_temp = behav.loc[behav['institution_code'] == i]
    sum_temp = pd.DataFrame(0, index=np.arange(len(feature_list)), columns=cat)
    sum_temp['institution'] = i 
    sum_temp['parameter'] = feature_list
    for t in feature_list:
        sum_temp.loc[sum_temp['parameter'] == t, 'weight'] = behav_temp[t].mean()
    summary_curves = pd.concat([summary_curves, sum_temp])
    
    
# Plot curve of predictors
tsummary_curves = pd.DataFrame()
tfeature_list =['100', '25', '12', '6', 'rchoice', 'uchoice', 'intercept']
cat = ['institution', 'weight', 'parameter']
for i in tbehav['institution_code'].unique():
    tbehav_temp = tbehav.loc[tbehav['institution_code'] == i]
    tsum_temp = pd.DataFrame(0, index=np.arange(len(feature_list)), columns=cat)
    tsum_temp['institution'] = i 
    tsum_temp['parameter'] = feature_list
    for t in feature_list:
        tsum_temp.loc[tsum_temp['parameter'] == t, 'weight'] = tbehav_temp[t].mean()
    tsummary_curves = pd.concat([tsummary_curves, tsum_temp])



# Visualization

figpath = figpath()

# Set seed for simulation
np.random.seed(1)



fig, ax =  plt.subplots(2,2, figsize = [10,10], sharey='row')
plt.sca(ax[0,0])
tsimulation = tbehav[tbehav['simulation_prob'].notnull()].copy()
tsimulation['signed_contrast'] = tsimulation['signed_contrast'].replace(-100, -35)
tsimulation['signed_contrast'] = tsimulation['signed_contrast'].replace(100, 35)
tsimulation = tsimulation.loc[tsimulation['signed_contrast'].abs() < 40]
sns.lineplot(data = tsimulation, x = 'signed_contrast', 
             y = 1*(tsimulation['choice']>0),
             legend = None, axes = ax[0,0], ci = 68, color='k')
tsimulation['simulation_run'] = np.random.binomial(1, p = tsimulation['simulation_prob'] )
sns.lineplot(data = tsimulation, x = 'signed_contrast', 
             y ='simulation_run', 
             legend = None, axes = ax[0,0], ci = 68, color='k')
ax[0,0].set_ylabel('Fraction of choices')
ax[0,0].set_ylim(0,1)
ax[0,0].set_xlabel('Signed contrast %')
ax[0,0].lines[1].set_linestyle("--")
ax[0,0].set_title('Level 1')

plt.sca(ax[1,0])
sns.lineplot(data = tsummary_curves, x = 'parameter', y = 'weight', 
             hue = 'institution', sort = False,  legend = None, palette = pal)
sns.despine()
labels = [item.get_text() for item in ax[1,0].get_xticklabels()]
labels[:] = ['100 %', '25 %', '12 %', '6 %', 'Rewarded choice (t-1)', 
             'Unrewarded choice (t-1)', 'Intercept']
ax[1,0].set_xticklabels(labels, rotation = 45, ha='right')
ax[1,0].set_ylabel('GLM weight')
ax[1,0].set_xlabel('')

plt.sca(ax[0,1])
bsimulation = behav[behav['simulation_prob'].notnull()].copy()
bsimulation['signed_contrast'] = bsimulation['signed_contrast'].replace(-100, -35)
bsimulation['signed_contrast'] = bsimulation['signed_contrast'].replace(100, 35)
bsimulation = bsimulation.loc[bsimulation['signed_contrast'].abs() < 40]
sns.lineplot(data = bsimulation, x = 'signed_contrast', 
             y = 1*(bsimulation['choice']>0), hue = 'probabilityLeft', 
             legend = None, axes = ax[0,1], ci = 68,  palette=['green', 'k','blue'])
bsimulation['simulation_run'] = np.random.binomial(1, p = bsimulation['simulation_prob'] )
sns.lineplot(data = bsimulation, x = 'signed_contrast', 
             y ='simulation_run', hue = 'probabilityLeft', 
             legend = None, axes = ax[0,1], ci = 68,  palette=['green', 'k','blue'])
ax[0,1].set_ylabel('Fraction of choices')
ax[0,1].set_ylim(0,1)
ax[0,1].set_xlabel('Signed contrast %')
ax[0,1].lines[3].set_linestyle("--")
ax[0,1].lines[4].set_linestyle("--")
ax[0,1].lines[5].set_linestyle("--")
ax[0,1].set_title('Level 2')

plt.sca(ax[1,1])
sns.lineplot(data = summary_curves, x = 'parameter', y = 'weight', 
             hue = 'institution', sort = False,  legend = None, palette = pal)
sns.despine()
labels = [item.get_text() for item in ax[1,0].get_xticklabels()]
labels[:] = ['100 %', '25 %', '12 %', '6 %', 'Rewarded choice (t-1)', 
             'Unrewarded choice (t-1)', 'Block Bias', 'Intercept']
ax[1,1].set_xticklabels(labels, rotation = 45, ha='right')
ax[1,1].set_ylabel('GLM weight')
ax[1,1].set_xlabel('')
plt.tight_layout()
fig.savefig(os.path.join(figpath, 'figure5_GLM.pdf'), dpi=600)

## Comparison of weights across labs

# Reduce size to just sessions per lab level 1
tbehav1 = tbehav.copy()
tbehav1 = tbehav1.groupby('subject_nickname').mean()
tbehav1['institution_code'] = \
    np.concatenate(tbehav.groupby('subject_nickname')['institution_code'].unique()).ravel().tolist() 

# Duplicate dataframe for all
learned_no_all = tbehav1.copy()
learned_no_all.loc[learned_no_all.shape[0] + 1, 'lab_number'] = 'All'
learned_2 = tbehav1.copy()
learned_2['institution_code'] = 'All'
learned_2 = tbehav1.append(learned_2)

# Set color palette
use_palette = [[0.6, 0.6, 0.6]] * len(np.unique(tbehav1['institution_code']))
use_palette = use_palette + [[1, 1, 0.2]]
sns.set_palette(use_palette)
lab_colors = group_colors()

# Plot behavioral metrics 
f, (ax1, ax2, ax3, ax4, ax5, ax6, ax7) = plt.subplots(1, 7, figsize=(20, 4))
sns.set_palette(lab_colors)

sns.swarmplot(y='100', x='institution_code', data=learned_no_all, hue='institution_code', ax=ax1)
axbox = sns.boxplot(y='100', x='institution_code', data=learned_2, color='white',
                    showfliers=False, ax=ax1)
ax1.set(ylabel='GLM weight: 100% contrast',  ylim=[-2, 5], xlabel='')
# [tick.set_color(lab_colors[i]) for i, tick in enumerate(ax1.get_xticklabels()[:-1])]
plt.setp(ax1.xaxis.get_majorticklabels(), rotation=40)
axbox.artists[-1].set_edgecolor('black')
for j in range(5 * (len(axbox.artists) - 1), 5 * len(axbox.artists)):
    axbox.lines[j].set_color('black')
ax1.get_legend().set_visible(False)

sns.swarmplot(y='25', x='institution_code', data=learned_no_all, hue='institution_code', ax=ax2)
axbox = sns.boxplot(y='25', x='institution_code', data=learned_2, color='white',
                    showfliers=False, ax=ax2)
ax2.set(ylabel='GLM weight: 25% contrast', ylim=[-2, 5], xlabel='')
# [tick.set_color(lab_colors[i]) for i, tick in enumerate(ax2.get_xticklabels()[:-1])]
plt.setp(ax2.xaxis.get_majorticklabels(), rotation=40)
axbox.artists[-1].set_edgecolor('black')
for j in range(5 * (len(axbox.artists) - 1), 5 * len(axbox.artists)):
    axbox.lines[j].set_color('black')
ax2.get_legend().set_visible(False)

sns.swarmplot(y='12', x='institution_code', data=learned_no_all, hue='institution_code', ax=ax3)
axbox = sns.boxplot(y='12', x='institution_code', data=learned_2, color='white', showfliers=False,
                    ax=ax3)
ax3.set(ylabel='GLM weight: 12% contrast', ylim=[-2, 5], xlabel='')
# [tick.set_color(lab_colors[i]) for i, tick in enumerate(ax3.get_xticklabels()[:-1])]
plt.setp(ax3.xaxis.get_majorticklabels(), rotation=40)
axbox.artists[-1].set_edgecolor('black')
for j in range(5 * (len(axbox.artists) - 1), 5 * len(axbox.artists)):
    axbox.lines[j].set_color('black')
ax3.get_legend().set_visible(False)

sns.swarmplot(y='6', x='institution_code', data=learned_no_all, hue='institution_code', ax=ax4)
axbox = sns.boxplot(y='6', x='institution_code', data=learned_2, color='white',
                    showfliers=False, ax=ax4)
ax4.set(ylabel='GLM weight: 6% contrast', ylim=[-2, 5], xlabel='')
# [tick.set_color(lab_colors[i]) for i, tick in enumerate(ax4.get_xticklabels()[:-1])]
plt.setp(ax4.xaxis.get_majorticklabels(), rotation=40)
axbox.artists[-1].set_edgecolor('black')
for j in range(5 * (len(axbox.artists) - 1), 5 * len(axbox.artists)):
    axbox.lines[j].set_color('black')
ax4.get_legend().set_visible(False)

sns.swarmplot(y='rchoice', x='institution_code', data=learned_no_all, hue='institution_code', ax=ax5)
axbox = sns.boxplot(y='rchoice', x='institution_code', data=learned_2, color='white',
                    showfliers=False, ax=ax5)
ax5.set(ylabel='GLM weight: Rewarded choice (t-1)',  ylim=[-2, 5], xlabel='')
# [tick.set_color(lab_colors[i]) for i, tick in enumerate(ax1.get_xticklabels()[:-1])]
plt.setp(ax5.xaxis.get_majorticklabels(), rotation=40)
axbox.artists[-1].set_edgecolor('black')
for j in range(5 * (len(axbox.artists) - 1), 5 * len(axbox.artists)):
    axbox.lines[j].set_color('black')
ax5.get_legend().set_visible(False)

sns.swarmplot(y='uchoice', x='institution_code', data=learned_no_all, hue='institution_code', ax=ax6)
axbox = sns.boxplot(y='uchoice', x='institution_code', data=learned_2, color='white',
                    showfliers=False, ax=ax6)
ax6.set(ylabel='GLM weight: Unrewarded choice (t-1)', ylim=[-2, 5], xlabel='')
# [tick.set_color(lab_colors[i]) for i, tick in enumerate(ax2.get_xticklabels()[:-1])]
plt.setp(ax6.xaxis.get_majorticklabels(), rotation=40)
axbox.artists[-1].set_edgecolor('black')
for j in range(5 * (len(axbox.artists) - 1), 5 * len(axbox.artists)):
    axbox.lines[j].set_color('black')
ax6.get_legend().set_visible(False)


sns.swarmplot(y='intercept', x='institution_code', data=learned_no_all, hue='institution_code', ax=ax7)
axbox = sns.boxplot(y='intercept', x='institution_code', data=learned_2, color='white', showfliers=False,
                    ax=ax7)
ax7.set(ylabel='GLM weight: Intercept', ylim=[-2, 5], xlabel='')
# [tick.set_color(lab_colors[i]) for i, tick in enumerate(ax3.get_xticklabels()[:-1])]
plt.setp(ax7.xaxis.get_majorticklabels(), rotation=40)
axbox.artists[-1].set_edgecolor('black')
for j in range(5 * (len(axbox.artists) - 1), 5 * len(axbox.artists)):
    axbox.lines[j].set_color('black')
ax7.get_legend().set_visible(False)
plt.tight_layout()

##########################################################################################################

# Reduce size to just sessions per lab level 1
tbehav1 = behav.copy()
tbehav1 = tbehav1.groupby('subject_nickname').mean()
tbehav1['institution_code'] = \
    np.concatenate(behav.groupby('subject_nickname')['institution_code'].unique()).ravel().tolist() 

# Duplicate dataframe for all
learned_no_all = tbehav1.copy()
learned_no_all.loc[learned_no_all.shape[0] + 1, 'lab_number'] = 'All'
learned_2 = tbehav1.copy()
learned_2['institution_code'] = 'All'
learned_2 = tbehav1.append(learned_2)

# Set color palette
use_palette = [[0.6, 0.6, 0.6]] * len(np.unique(tbehav1['institution_code']))
use_palette = use_palette + [[1, 1, 0.2]]
sns.set_palette(use_palette)
lab_colors = group_colors()

# Plot behavioral metrics 
f, (ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8) = plt.subplots(1, 8, figsize=(20, 4))
sns.set_palette(lab_colors)

sns.swarmplot(y='100', x='institution_code', data=learned_no_all, hue='institution_code', ax=ax1)
axbox = sns.boxplot(y='100', x='institution_code', data=learned_2, color='white',
                    showfliers=False, ax=ax1)
ax1.set(ylabel='GLM weight: 100% contrast',  ylim=[-2, 6], xlabel='')
# [tick.set_color(lab_colors[i]) for i, tick in enumerate(ax1.get_xticklabels()[:-1])]
plt.setp(ax1.xaxis.get_majorticklabels(), rotation=40)
axbox.artists[-1].set_edgecolor('black')
for j in range(5 * (len(axbox.artists) - 1), 5 * len(axbox.artists)):
    axbox.lines[j].set_color('black')
ax1.get_legend().set_visible(False)

sns.swarmplot(y='25', x='institution_code', data=learned_no_all, hue='institution_code', ax=ax2)
axbox = sns.boxplot(y='25', x='institution_code', data=learned_2, color='white',
                    showfliers=False, ax=ax2)
ax2.set(ylabel='GLM weight: 25% contrast', ylim=[-2, 6], xlabel='')
# [tick.set_color(lab_colors[i]) for i, tick in enumerate(ax2.get_xticklabels()[:-1])]
plt.setp(ax2.xaxis.get_majorticklabels(), rotation=40)
axbox.artists[-1].set_edgecolor('black')
for j in range(5 * (len(axbox.artists) - 1), 5 * len(axbox.artists)):
    axbox.lines[j].set_color('black')
ax2.get_legend().set_visible(False)

sns.swarmplot(y='12', x='institution_code', data=learned_no_all, hue='institution_code', ax=ax3)
axbox = sns.boxplot(y='12', x='institution_code', data=learned_2, color='white', showfliers=False,
                    ax=ax3)
ax3.set(ylabel='GLM weight: 12% contrast', ylim=[-2, 6], xlabel='')
# [tick.set_color(lab_colors[i]) for i, tick in enumerate(ax3.get_xticklabels()[:-1])]
plt.setp(ax3.xaxis.get_majorticklabels(), rotation=40)
axbox.artists[-1].set_edgecolor('black')
for j in range(5 * (len(axbox.artists) - 1), 5 * len(axbox.artists)):
    axbox.lines[j].set_color('black')
ax3.get_legend().set_visible(False)

sns.swarmplot(y='6', x='institution_code', data=learned_no_all, hue='institution_code', ax=ax4)
axbox = sns.boxplot(y='6', x='institution_code', data=learned_2, color='white',
                    showfliers=False, ax=ax4)
ax4.set(ylabel='GLM weight: 6% contrast', ylim=[-2, 6], xlabel='')
# [tick.set_color(lab_colors[i]) for i, tick in enumerate(ax4.get_xticklabels()[:-1])]
plt.setp(ax4.xaxis.get_majorticklabels(), rotation=40)
axbox.artists[-1].set_edgecolor('black')
for j in range(5 * (len(axbox.artists) - 1), 5 * len(axbox.artists)):
    axbox.lines[j].set_color('black')
ax4.get_legend().set_visible(False)

sns.swarmplot(y='rchoice', x='institution_code', data=learned_no_all, hue='institution_code', ax=ax5)
axbox = sns.boxplot(y='rchoice', x='institution_code', data=learned_2, color='white',
                    showfliers=False, ax=ax5)
ax5.set(ylabel='GLM weight: Rewarded choice (t-1)',  ylim=[-2, 6], xlabel='')
# [tick.set_color(lab_colors[i]) for i, tick in enumerate(ax1.get_xticklabels()[:-1])]
plt.setp(ax5.xaxis.get_majorticklabels(), rotation=40)
axbox.artists[-1].set_edgecolor('black')
for j in range(5 * (len(axbox.artists) - 1), 5 * len(axbox.artists)):
    axbox.lines[j].set_color('black')
ax5.get_legend().set_visible(False)

sns.swarmplot(y='uchoice', x='institution_code', data=learned_no_all, hue='institution_code', ax=ax6)
axbox = sns.boxplot(y='uchoice', x='institution_code', data=learned_2, color='white',
                    showfliers=False, ax=ax6)
ax6.set(ylabel='GLM weight: Unrewarded choice (t-1)', ylim=[-2, 6], xlabel='')
# [tick.set_color(lab_colors[i]) for i, tick in enumerate(ax2.get_xticklabels()[:-1])]
plt.setp(ax6.xaxis.get_majorticklabels(), rotation=40)
axbox.artists[-1].set_edgecolor('black')
for j in range(5 * (len(axbox.artists) - 1), 5 * len(axbox.artists)):
    axbox.lines[j].set_color('black')
ax6.get_legend().set_visible(False)

sns.swarmplot(y='block', x='institution_code', data=learned_no_all, hue='institution_code', ax=ax7)
axbox = sns.boxplot(y='block', x='institution_code', data=learned_2, color='white', showfliers=False,
                    ax=ax7)
ax7.set(ylabel='GLM weight: Block Bias', ylim=[-2, 6], xlabel='')
# [tick.set_color(lab_colors[i]) for i, tick in enumerate(ax3.get_xticklabels()[:-1])]
plt.setp(ax7.xaxis.get_majorticklabels(), rotation=40)
axbox.artists[-1].set_edgecolor('black')
for j in range(5 * (len(axbox.artists) - 1), 5 * len(axbox.artists)):
    axbox.lines[j].set_color('black')
ax7.get_legend().set_visible(False)
plt.tight_layout()

sns.swarmplot(y='intercept', x='institution_code', data=learned_no_all, hue='institution_code', ax=ax8)
axbox = sns.boxplot(y='intercept', x='institution_code', data=learned_2, color='white', showfliers=False,
                    ax=ax8)
ax8.set(ylabel='GLM weight: Intercept', ylim=[-2, 6], xlabel='')
# [tick.set_color(lab_colors[i]) for i, tick in enumerate(ax3.get_xticklabels()[:-1])]
plt.setp(ax8.xaxis.get_majorticklabels(), rotation=40)
axbox.artists[-1].set_edgecolor('black')
for j in range(5 * (len(axbox.artists) - 1), 5 * len(axbox.artists)):
    axbox.lines[j].set_color('black')
ax8.get_legend().set_visible(False)
plt.tight_layout()
sns.despine()

# Make common dataframe for Level 1 to level 2

tbehav1 = tbehav.copy()
tbehav1 = tbehav1.groupby('subject_nickname').mean()
tbehav1['institution_code'] = \
    np.concatenate(tbehav.groupby('subject_nickname')['institution_code'].unique()).ravel().tolist() 

sbehav1 = behav.copy()
sbehav1 = sbehav1.groupby('subject_nickname').mean()
sbehav1['institution_code'] = \
    np.concatenate(behav.groupby('subject_nickname')['institution_code'].unique()).ravel().tolist() 


blab = behav.groupby('institution_code').mean()
sns.scatterplot(data=tbehav1, x='rchoice', y='rchoice')