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
from paper_behavior_functions import (seaborn_style, query_sessions_around_criterion, seaborn_style,
                                      institution_map, group_colors, figpath, EXAMPLE_MOUSE)
from dj_tools import dj2pandas, fit_psychfunc
from ibl_pipeline import behavior, subject, reference, acquisition
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score, KFold
import os
from ibl_pipeline.utils import psychofit as psy
from scipy import stats

# whether to query data from DataJoint (True), or load from disk (False)
query = True

##############################################################################
#******************************* Functions **********************************#
##############################################################################


def model_psychometric_history(behav):
    select =  behav.copy()
    select['t-1'] = select['trial_feedback_type'].shift(periods=1).to_numpy()
    select.loc[select['choice'] == -1, 'choice'] = 0 
    select = select.iloc[1:,:]
    #select['t-1'].fillna(0,  inplace=True)
    select['t-1']  = select['t-1'].astype(int)
    
    plot_psychometric(select.loc[select['signed_contrast'],
                     select.loc[select['probabilityLeft'] ==i, 'signed_contrast']], 
                     palette = ['red', 'green'], 
                     ci = 68)

    sns.lineplot(data = select_50, hue = 't-1', x = select_50['signed_contrast'],
                     y = select_50['simulation_prob'], palette = ['red', 'green'], ci = 68)


def run_glm(behav, example, correction = True,  bias = False, cross_validation = True):
    for i, nickname in enumerate(np.unique(behav['subject_nickname'])):
        if np.mod(i+1, 10) == 0:
            print('Loading data of subject %d of %d' % (i+1, len(
                    np.unique(behav['subject_nickname']))))
    
        # Get the trials of the sessions around criterion
        trials = behav.loc[behav['subject_nickname'] == nickname].copy()
        
        
        if bias == True:
            neutral_n = fit_psychfunc(behav[(behav['subject_nickname'] == nickname)
                                   & (behav['probabilityLeft'] == 50)])
            left_fit = fit_psychfunc(behav[(behav['subject_nickname'] == nickname)
                                           & (behav['probabilityLeft'] == 80)])
            right_fit = fit_psychfunc(behav[(behav['subject_nickname'] == nickname)
                                            & (behav['probabilityLeft'] == 20)])
            
            behav.loc[behav['subject_nickname'] == nickname, 'bias_n'] = \
                neutral_n.loc[0, 'bias']
            behav.loc[behav['subject_nickname'] == nickname, 'bias_r'] = \
                right_fit.loc[0, 'bias']
            behav.loc[behav['subject_nickname'] == nickname, 'bias_l'] = \
                left_fit.loc[0, 'bias']
    
        else:
            fit_df = dj2pandas(trials.copy())
            fit_result = fit_psychfunc(fit_df)
            behav.loc[behav['subject_nickname'] == nickname, 'threshold'] = \
                fit_result.loc[0, 'threshold']
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
        
        # Apply correction
        if correction == True:
           data['rchoice+1'] = \
           data['rchoice'].shift(periods=-1).to_numpy()
           data['uchoice+1'] = \
           data['uchoice'].shift(periods=-1).to_numpy()
            
        # Shift rewarded and unrewarded predictors by one
        data.loc[:, ['rchoice', 'uchoice']] = \
            data[['rchoice', 'uchoice']].shift(periods=1).to_numpy()
            
    
        # Drop any nan trials
        data.dropna(inplace=True)
        
        # Make sensory predictors (no 0 predictor)
        contrasts = [ 25, 100,  12.5,   6.25]
        for i in contrasts:
            data.loc[(data['signed_contrast'].abs() == i), i] = \
                np.sign(data.loc[(data['signed_contrast'].abs() == i),
                                 'signed_contrast'].to_numpy())
            
            data_con =  data[i].copy()
            data[i] = data_con.fillna(0)
        
        # If contrast missing break
        for i in contrasts:
            if np.sum(data[i]) == 0:
                print('missing contrast')
                missing_contrast = True
            else:
                missing_contrast = False
        
        if missing_contrast == True:
            continue
        
        # Make block identity (across predictors right is positive, hence logic below)
        if bias == True:
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
        
        if cross_validation == False:
            X_train = exog.copy()
            X_test = exog.copy()
            y_train = endog.copy()
            y_test = endog.copy()
            
        else:
            X_train = exog.iloc[:int(len(exog)*0.70),:].copy()
            X_test = exog.iloc[int(len(endog)*0.70):,:].copy()
            y_train = endog.iloc[:int(len(endog)*0.70)].copy()
            y_test = endog.iloc[int(len(endog)*0.70):].copy()
        
        # Store index
        
        index = X_test['index'].to_numpy()
        X_train.drop(columns=['index'], inplace=True)
        X_test.drop(columns=['index'], inplace=True)
        
        
        # Fit model
        try:
            logit_model = sm.Logit(y_train, X_train)
            result = logit_model.fit_regularized()
            # print(result.summary2())
            
            # Store model weights
            behav.loc[behav['subject_nickname'] == nickname, 'intercept'] = result.params['const'].copy()
            behav.loc[behav['subject_nickname'] == nickname, 'rchoice'] = result.params['rchoice'].copy()
            behav.loc[behav['subject_nickname'] == nickname, 'uchoice'] = result.params['uchoice'].copy()
            mask = result.params.index.get_level_values(0)
            behav.loc[behav['subject_nickname'] == nickname, '25'] = result.params[25].copy()
            behav.loc[behav['subject_nickname'] == nickname, '6'] = result.params.loc[mask == 6.25][0]
            behav.loc[behav['subject_nickname'] == nickname, '100'] = result.params[100].copy()
            behav.loc[behav['subject_nickname'] == nickname, '12'] = result.params.loc[mask == 12.5][0]
            
            if bias == True:
                behav.loc[behav['subject_nickname'] == nickname, 'block'] = result.params['block'].copy()
            
            if correction == True:
                result.params['rchoice+1'] = result.params['rchoice+1'] * -1
                result.params['uchoice+1'] = result.params['uchoice+1'] * -1
                behav.loc[behav['subject_nickname'] == nickname, 'rchoice+1'] = result.params['rchoice+1'].copy()
                behav.loc[behav['subject_nickname'] == nickname, 'uchoice+1'] = result.params['uchoice+1'].copy()
            # Probabilities on test data
            prob = result.predict(X_test).to_numpy()
            
            if nickname == example:
                example_model = result
            
            # Propagate to storing dataframe
            behav.loc[behav['index'].isin(index), 'simulation_prob'] = prob
        except:
            print('singular matrix')
    return behav, example_model


def data_2_X_test (behav, correction = True, bias = True):
        data = behav[['index','trial_feedback_type',
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
        
        # Apply correction
        if correction == True:
           data['rchoice+1'] = \
           data['rchoice'].shift(periods=-1).to_numpy()
           data['uchoice+1'] = \
           data['uchoice'].shift(periods=-1).to_numpy()
            
        # Shift rewarded and unrewarded predictors by one
        data.loc[:, ['rchoice', 'uchoice']] = \
            data[['rchoice', 'uchoice']].shift(periods=1).to_numpy()
            
    
        # Drop any nan trials
        data.dropna(inplace=True)
        
        # Make sensory predictors (no 0 predictor)
        contrasts = [ 25, 100,  12.5,   6.25]
        for i in contrasts:
            data.loc[(data['signed_contrast'].abs() == i), i] = \
                np.sign(data.loc[(data['signed_contrast'].abs() == i),
                                 'signed_contrast'].to_numpy())
            
            data_con =  data[i].copy()
            data[i] = data_con.fillna(0)
        
        # Make block identity (across predictors right is positive, hence logic below)
        if bias == True:
            data.loc[(data['probabilityLeft'] == 50), 'block'] = 0
            data.loc[(data['probabilityLeft'] == 20), 'block'] = 1
            data.loc[(data['probabilityLeft'] == 80), 'block'] = -1
        
        # Make choice in between 0 and 1 -> 1 for right and 0 for left
        data.loc[data['choice'] == -1, 'choice'] = 0
        
        index = data['index'].copy()

        # Create predictor matrix
        endog = data['choice'].copy()
        exog = data.copy()
        exog.drop(columns=['index', 'trial_feedback_type', 
                       'signed_contrast', 'choice', 
                           'probabilityLeft'], inplace=True)
        exog = sm.add_constant(exog)
        
        return exog, index


def plot_psychometric(x, y, col, point = False, mark = 'o', al =1):
    # summary stats - average psychfunc over observers
    df = pd.DataFrame({'signed_contrast': x, 'choice': y,
                       'choice2': y})
    df2 = df.groupby(['signed_contrast']).agg(
        {'choice2': 'count', 'choice': 'mean'}).reset_index()
    df2.rename(columns={"choice2": "ntrials",
                        "choice": "fraction"}, inplace=True)
    df2 = df2.groupby(['signed_contrast']).mean().reset_index()
    df2 = df2[['signed_contrast', 'ntrials', 'fraction']]

    # fit psychfunc
    pars, L = psy.mle_fit_psycho(df2.transpose().values,  # extract the data from the df
                                 P_model='erf_psycho_2gammas',
                                 parstart=np.array(
                                     [df2['signed_contrast'].mean(), 20., 0.05, 0.05]),
                                 parmin=np.array(
                                     [df2['signed_contrast'].min(), 5, 0., 0.]),
                                 parmax=np.array([df2['signed_contrast'].max(), 100., 1, 1]))

    # plot psychfunc
    g = sns.lineplot(np.arange(-29, 29),
                     psy.erf_psycho_2gammas(pars, np.arange(-29, 29)), color = col,  
                     alpha = al)

    # plot psychfunc: -100, +100
    sns.lineplot(np.arange(-37, -32),
                 psy.erf_psycho_2gammas(pars, np.arange(-103, -98)), color = col,
                 alpha = al)
    sns.lineplot(np.arange(32, 37),
                 psy.erf_psycho_2gammas(pars, np.arange(98, 103)), color = col,  
                 alpha = al)

    # now break the x-axis
    # if 100 in df.signed_contrast.values and not 50 in
    # df.signed_contrast.values:
    df['signed_contrast'] = df['signed_contrast'].replace(-100, -35)
    df['signed_contrast'] = df['signed_contrast'].replace(100, 35)
    
    if point == True:
        sns.lineplot(df['signed_contrast'], df['choice'], err_style="bars",
                         linewidth=0, linestyle='None', mew=0.5,
                         marker=mark, ci=95, color = col, alpha = al)

    g.set_xticks([-35, -25, -12.5, 0, 12.5, 25, 35])
    g.set_xticklabels(['-100', '-25', '-12.5', '0', '12.5', '25', '100'],
                      size='small', rotation=45)
    g.set_xlim([-40, 40])
    g.set_ylim([0, 1])
    g.set_yticks([0, 0.25, 0.5, 0.75, 1])
    g.set_yticklabels(['0', '25', '50', '75', '100'])

##############################################################################
#*******************************Biased Task**********************************#
##############################################################################

# Set properties of the analysis '2019-09-20 10:15:34'
bsession= '2019-08-31 11:59:37'
tsession= '2019-08-23 11:00:32'
correction = False

if query is True:
    # Query sessions biased data 
    use_sessions, _ = query_sessions_around_criterion(criterion='biased',
                                                      days_from_criterion=[2, 3])
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
    behav_merged = dj2pandas(bdat)
    behav_merged['institution_code'] = behav_merged.institution_short.map(institution_map)
else:
    behav_merged = pd.read_csv(join('data', 'Fig5.csv'))

# Variable to store model parameters
behav_merged['rchoice'] = np.nan
behav_merged['uchoice'] = np.nan
behav_merged['6'] = np.nan
behav_merged['12'] = np.nan
behav_merged['25'] = np.nan
behav_merged['100'] = np.nan
behav_merged['block'] = np.nan
behav_merged['intercept'] = np.nan
behav_merged['simulation_prob'] = np.nan

if correction == True:
    behav_merged['rchoice+1'] = np.nan
    behav_merged['uchoice+1'] = np.nan
    behav_merged['choice+1'] = np.nan


# Drop trials with weird contrasts
behav_merged.drop(behav_merged['probabilityLeft']
                  [~behav_merged['probabilityLeft'].isin([50,20,80])].index,
        inplace=True)
behav_merged.drop(behav_merged['probabilityLeft']
                  [~behav_merged['signed_contrast'].isin(
                      [100,25,12.5,6.25,0,-6.25,-12.5,-25,-100])].index, 
                          inplace=True)

# split the two types of task protocols (remove the pybpod version number
behav_merged['task'] = behav_merged['task_protocol'].str[14:20].copy()

behav = behav_merged.loc[behav_merged['task']=='biased'].copy() 
behav = behav.reset_index()

behav, example_model = run_glm(behav, EXAMPLE_MOUSE, correction = correction,
                               bias = True, cross_validation  = False)

        
##############################################################################
#*****************************Unbiased Task**********************************#
##############################################################################    

# Query sessions traning data 
tbehav = behav_merged.loc[behav_merged['task']=='traini'].copy()
tbehav.drop(tbehav['probabilityLeft'][~tbehav['probabilityLeft'].isin([50])].index,
        inplace=True)
tbehav = tbehav.reset_index()

tbehav , example_model_t = run_glm(tbehav, EXAMPLE_MOUSE, correction = correction,
                                   bias = False, cross_validation  = False)



##############################################################################
#*****************************Summary information****************************#
##############################################################################    


# Plot curve of predictors
summary_curves = pd.DataFrame()

if correction == True:
    feature_list =['100', '25', '12', '6', 'rchoice', 'rchoice+1', 'uchoice',
                   'uchoice+1', 'block', 'intercept']
else:
    feature_list =['100', '25', '12', '6', 'rchoice', 'uchoice', 
                    'block', 'intercept']


cat = ['institution', 'weight', 'parameter']
for i in behav['institution_code'].unique():
    behav_temp = behav.loc[behav['institution_code'] == i]
    sum_temp = pd.DataFrame(0, index=np.arange(len(feature_list)), columns=cat)
    sum_temp['institution'] = i 
    sum_temp['parameter'] = feature_list
    for t in feature_list:
        sum_temp.loc[sum_temp['parameter'] == t, 'weight'] = \
            behav_temp[t].mean()
    summary_curves = pd.concat([summary_curves, sum_temp])
    
    
# Plot curve of predictors
tsummary_curves = pd.DataFrame()
tfeature_list =['100', '25', '12', '6', 'rchoice', 'uchoice',
                 'intercept']
cat = ['institution', 'weight', 'parameter']
for i in tbehav['institution_code'].unique():
    tbehav_temp = tbehav.loc[tbehav['institution_code'] == i]
    tsum_temp = pd.DataFrame(0, index=np.arange(len(tfeature_list)), 
                             columns=cat)
    tsum_temp['institution'] = i 
    tsum_temp['parameter'] = tfeature_list
    for t in tfeature_list:
        tsum_temp.loc[tsum_temp['parameter'] == t, 'weight'] = \
            tbehav_temp[t].mean()
    tsummary_curves = pd.concat([tsummary_curves, tsum_temp])

##############################################################################
#******************************* Plotting ***********************************#
##############################################################################

pal = group_colors()

# Visualization

figpath = figpath()

# Set seed for simulation
np.random.seed(1)

# Line colors
cmap = sns.diverging_palette(20, 220, n=3, center="dark")

# Get data from example session (TODO make into specific query)


# Data for bias session

b = (subject.Subject * behavior.TrialSet.Trial * acquisition.Session
  & 'subject_nickname="%s"' % EXAMPLE_MOUSE & 'task_protocol LIKE "%biased%"')

bdat = b.fetch(order_by='session_start_time, trial_id',
               format='frame').reset_index()

ebehav = dj2pandas(bdat)
ebehav = ebehav.reset_index()
bebehav = ebehav.loc[ebehav['subject_nickname'] ==  EXAMPLE_MOUSE]
bebehav_model_data, index = data_2_X_test (bebehav, correction = \
                                           correction, bias = True)
bebehav.loc[bebehav['index'].isin(index) ,'simulation_prob'] = \
    example_model.predict(bebehav_model_data).to_numpy()# Run simulation



# Data for examples
use_sessions, use_days = query_sessions_around_criterion(criterion='ephys',
                                                         days_from_criterion=[
                                                            2, 0],
                                                         as_dataframe=False)
b = (use_sessions * subject.Subject * subject.SubjectLab * reference.Lab
     * behavior.TrialSet.Trial)

b2 = b.proj('institution_short', 'subject_nickname', 'task_protocol',
            'trial_stim_contrast_left', 'trial_stim_contrast_right', 
            'trial_response_choice', 'task_protocol', 'trial_stim_prob_left', 
            'trial_feedback_type')
bdat1 = b2.fetch(order_by=
        'institution_short, subject_nickname, session_start_time, trial_id',
                format='frame').reset_index()

tbehav = dj2pandas(bdat1)
tbehav = tbehav.reset_index()
tebehav = tbehav.loc[tbehav['subject_nickname'] ==  EXAMPLE_MOUSE]
tebehav_model_data, index = data_2_X_test (tebehav, correction = correction,
                                           bias = False)
tebehav.loc[tebehav['index'].isin(index) ,'simulation_prob'] = \
    example_model_t.predict(tebehav_model_data).to_numpy()# Run simulation



# Run simulation
simulation_size = 1
tsimulation = tebehav[tebehav['simulation_prob'].notnull()].copy()
tsimulation = tsimulation[tsimulation['subject_nickname'] == EXAMPLE_MOUSE].copy()

rsimulation = pd.concat([tsimulation]*simulation_size)
rsimulation['simulation_run'] = \
    np.random.binomial(1, p = rsimulation['simulation_prob'])
bsimulation = bebehav[bebehav['simulation_prob'].notnull()].copy()
bsimulation = \
    bsimulation[bsimulation['subject_nickname'] == EXAMPLE_MOUSE].copy()
brsimulation = pd.concat([bsimulation]*simulation_size)
brsimulation['simulation_run'] = \
    np.random.binomial(1, p = brsimulation['simulation_prob'])


# Figure of single session

fig, ax =  plt.subplots(1,2, figsize = [10,5], sharey='row')
plt.sca(ax[0])

# Drop 50s, they are only a temporary step
rsimulation = rsimulation[abs(rsimulation['signed_contrast'])!= 50]
tsimulation = tsimulation[abs(tsimulation['signed_contrast'])!= 50]

# Plot
seaborn_style()

sns.lineplot(rsimulation['signed_contrast'], rsimulation['simulation_run'], 
             color = 'k', ci = 95, linewidth = 0)
  
plot_psychometric(tsimulation['signed_contrast'], 
  tsimulation['choice_right'], 'k', point = True, mark = 'o', al = 0.5)
ax[0].lines[1].set_alpha(0)
ax[0].lines[2].set_alpha(0)
ax[0].lines[3].set_alpha(0)
ax[0].set_ylabel('Fraction of choices')
ax[0].set_ylim(0,1)
ax[0].set_xlabel('Signed contrast %')
ax[0].set_title('Basic task - Example mouse')
plt.sca(ax[1])
for c, i  in enumerate([20, 50, 80]):
    subset = brsimulation.loc[brsimulation['probabilityLeft'] == i]
    subset1 = bsimulation.loc[bsimulation['probabilityLeft'] == i]
    sns.lineplot(subset['signed_contrast'], subset['simulation_run'], ci = 95,
                 color =cmap[c], linewidth = 0)
    #plot_psychometric(subset['signed_contrast'], 
    #                  subset['simulation_run'], col = cmap[c] ,  point = True,  mark = '^')
    plot_psychometric(subset1['signed_contrast'], 
                      subset1['choice_right'], cmap[c] , point = True,  mark = 'o', al = 0.5)

ax[1].lines[0].set_alpha(0)
ax[1].lines[1].set_alpha(0)
ax[1].lines[3].set_alpha(0)
ax[1].lines[7].set_alpha(0)
ax[1].lines[8].set_alpha(0)
ax[1].lines[9].set_alpha(0)
ax[1].lines[11].set_alpha(0)
ax[1].lines[12].set_alpha(0)
ax[1].lines[13].set_alpha(0)

    
ax[1].set_ylabel('Fraction of choices')
ax[1].set_ylim(0,1)
ax[1].set_xlabel('Signed contrast %')
ax[1].set_title('Full task - Example mouse')
sns.despine(trim=True)
plt.tight_layout()
fig.savefig(os.path.join(figpath, 'figure5_GLM.pdf'), dpi=600)


# Biased Weights

fig, ax  = plt.subplots(1,3, figsize = [10,5])
plt.sca(ax[0])
bsensory = summary_curves[summary_curves['parameter'].isin(['6','25','12', '100'])]
sns.swarmplot(data = bsensory, hue = 'institution', x = 'parameter', y= 'weight', 
             palette= pal, order=['6','12','25','100'], size = 7)
ax[0].get_legend().set_visible(False)
ax[0].set_xlabel('Fitted Visual Parameter (Contrast %)')
ax[0].set_ylabel('Weight')
ax[0].set_ylim(0,5)
plt.sca(ax[1])
breward= summary_curves[summary_curves['parameter'].isin(['rchoice','uchoice'])]
sns.swarmplot(data = breward, hue = 'institution', x = 'parameter', y= 'weight', 
             palette= pal, order=['rchoice','uchoice'], size = 7)
ax[1].get_legend().set_visible(False)
ax[1].set_xlabel(' ')
if correction == True:
    ax[1].set_xticklabels(['Corrected Rewarded Choice (t-1)', 
                           'Corrected Unrewarded Choice (t-1)'], rotation = 45, ha='right')
else:
    ax[1].set_xticklabels(['Rewarded \n Choice (t-1)', 'Unrewarded \n Choice (t-1)'],
                          ha='center')
ax[1].set_ylabel('Weight')
ax[1].set_ylim(0,1)
plt.sca(ax[2])
bbias= summary_curves[summary_curves['parameter'].isin(['block', 'intercept'])]
sns.swarmplot(data = bbias, hue = 'institution', x = 'parameter', y= 'weight', 
             palette= pal,  size = 7, order =
             ['block', 'intercept'])
ax[2].get_legend().set_visible(False)
ax[2].set_xlabel(' ')
ax[2].set_xticklabels(['Bias', 'Block Bias'], rotation = 45, ha='right')
ax[2].set_ylabel('Weight')
ax[2].set_ylim(-1,1)

plt.tight_layout()
sns.despine(trim=True)
fig.savefig(os.path.join(figpath, 'figure5_biased _weights.pdf'), dpi=600)


# Unbiased Weights
fig, ax  = plt.subplots(1,3, figsize = [10,5])
plt.sca(ax[0])
bsensory = tsummary_curves[tsummary_curves['parameter'].isin(['6','25','12', '100'])]
sns.swarmplot(data = bsensory, hue = 'institution', x = 'parameter', y= 'weight', 
             palette= pal, order=['6','12','25','100'], size = 7)
ax[0].get_legend().set_visible(False)
ax[0].set_xlabel('Fitted Visual Parameter (Contrast %)')
ax[0].set_ylabel('Weight')
ax[0].set_ylim(0,5)

plt.sca(ax[1])
breward= tsummary_curves[tsummary_curves['parameter'].isin(['rchoice','uchoice'])]
sns.swarmplot(data = breward, hue = 'institution', x = 'parameter', y= 'weight', 
             palette= pal, order=['rchoice','uchoice', 'rchoice+1','uchoice+1'], size = 7)
ax[1].get_legend().set_visible(False)
ax[1].set_xlabel(' ')
ax[1].set_ylim(0,1)

if correction == True:
    ax[1].set_xticklabels(['Corrected Rewarded \n Choice (t-1)', 
                           'Corrected Unrewarded \n Choice (t-1)'], 
                           ha='right')
else:
     ax[1].set_xticklabels(['Rewarded \n Choice (t-1)', 
                            'Unrewarded \n Choice (t-1)'], 
                           ha='center')
ax[1].set_ylabel('Weight')
plt.sca(ax[2])
bbias= tsummary_curves[tsummary_curves['parameter'].isin(['block', 'intercept'])]
sns.swarmplot(data = bbias, hue = 'institution', x = 'parameter', y= 'weight', 
             palette= pal, order=['intercept'], size = 7)
ax[2].get_legend().set_visible(False)
ax[2].set_xlabel('     ')
ax[2].set_xticklabels(['Bias'], ha='center')
ax[2].set_ylabel('Weight')
ax[2].set_ylim(-1,1)
plt.tight_layout()
sns.despine(trim=True)
fig.savefig(os.path.join(figpath, 'figure5_unbiased _weights.pdf'), dpi=600)


# Accuracy
behav_merge = pd.concat([behav, tbehav]).copy()
behav_merge = behav_merge[~np.isnan(behav_merge['simulation_prob'])]
behav_merge['accu'] = behav_merge['simulation_prob'].round().to_numpy()
behav_merge.loc[behav_merge['accu'] == 0, 'accu']  = -1
behav_merge['accu'] =  (behav_merge['accu'] == behav_merge['choice'])
behav_merge1 =  behav_merge.copy()
behav_merge1['institution_code'] = 'All'
behav_merge = pd.concat([behav_merge, behav_merge1])

grouped = behav_merge.groupby(['subject_nickname',
                         'institution_code'])['accu'].mean().reset_index()
pal.append((1,1,0))

fig, ax  = plt.subplots()
plt.sca(ax)
sns.boxplot(data = grouped, x = 'institution_code', y =grouped['accu']*100, palette= pal, 
            order= ['Lab 1','Lab 2','Lab 3','Lab 4','Lab 5','Lab 6','Lab 7',
                    'All'], linewidth = 2)
sns.swarmplot(data=grouped, x = 'institution_code', y =grouped['accu']*100,color = 'k', 
              order=['Lab 1','Lab 2','Lab 3','Lab 4','Lab 5','Lab 6','Lab 7',
                    'All'])
ax.set_xlabel('Laboratory')
ax.set_ylabel('Model Accuracy %')
sns.despine(trim=True)
fig.savefig(os.path.join(figpath, 'figure5_accuracy.pdf'), dpi=600)




    