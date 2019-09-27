#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 10 01:49:23 2019
TODO : Divide by sex of the animal
prepare ibl trial dataframe for IBL
@author: ibladmin
"""

import matplotlib.pyplot as plt
import pandas as pd
## CONNECT TO datajoint
import datajoint as dj
dj.config['database.host'] = 'datajoint.internationalbrainlab.org'
from ibl_pipeline.analyses import behavior as behavior_analysis
from ibl_pipeline import reference, subject, behavior
from alexfigs_datajoint_functions import *  
import seaborn as sns
from paper_behavior_functions import figpath
import os

save_path = figpath()  # Path to where figures are saved
key = ((subject.Subject()  & 'sex!="U"') * (behavior.TrialSet() & 'n_trials > 100') * (subject.SubjectLab()) * (behavior_analysis.SessionTrainingStatus() & 'training_status="ready for ephys"  ')).fetch('KEY')
trials_ibl = pd.DataFrame.from_dict((subject.Subject() * behavior.TrialSet.Trial & key).fetch(as_dict=True))

trials_ibl['signed_contrasts'] = trials_ibl['trial_stim_contrast_right'] - trials_ibl['trial_stim_contrast_left']

##Rename for GLM function
trials_ibl = trials_ibl.rename(index=str, columns={"session_start_time": "ses", 
                                      "subject_uuid": "mouse_name", 
                                      "trial_feedback_type": "feedbackType", 
                                      "trial_response_choice":"choice"})

#Rename choices
trials_ibl.loc[(trials_ibl['choice']=='CW'),'choice'] = -1
trials_ibl.loc[(trials_ibl['choice']=='CCW'), 'choice'] = 1
trials_ibl.loc[(trials_ibl['choice']=='No Go'), 'choice'] = 0

#Remove 0.5 block?


psy_df =  trials_ibl.loc[(trials_ibl['trial_stim_prob_left'] == 0.8) | (trials_ibl['trial_stim_prob_left'] == 0.2)]

mresult, fresult, result, mr2, fr2,r2  = glm_logit(psy_df)

mresults  =  pd.DataFrame({"Predictors": mresult.model.exog_names , "Coef" : mresult.params.values,\
                          "SEM": mresult.bse.values, "Sex": "M"})
fresults  =  pd.DataFrame({"Predictors": fresult.model.exog_names , "Coef" : fresult.params.values,\
                          "SEM": mresult.bse.values, "Sex": "F"}).reindex(mresults.Coef.abs().sort_values().index)
results  = pd.concat([mresults, fresults]) 


#Plotting
fig, ax = plt.subplots(figsize=(12, 9))
ax  = sns.barplot(x = 'Predictors', y = 'Coef', data=results, hue='Sex')    
ax.set_xticklabels( results['Predictors'], rotation=-90)
ax.set_ylabel('coef')
ax.axhline(y=0, linestyle='--', color='black', linewidth=2)
fig.suptitle ('GLM Biased Blocks')
fig.savefig(os.path.join(save_path, "glm_sex_diff.pdf"))

#Plotting pooled results
plot_glm(psy_df, result, r2)


## GLM functions
"""
Created on Tue Jun  4 16:55:17 2019


Predictors:  Rewarded choice and unrewarded choice
These two predictors include  correct choice and reward i.e

R (Right_Choice = True, Correct_choice = True, Reward = True) = 1
R (Right_Choice = True, Correct_choice = False, Reward = True) = 1 - This never happens
U(Right_Choice = True, Correct_choice = False, Reward = False) = 0
U(Right_Choice = True, Correct_choice = True, Reward = False) = 0

R (Right_Choice = False, Correct_choice = True, Reward = True) =  -1
R (Right_Choice = False, Correct_choice = False, Reward = True) =  -1 - This never happens
U(Right_Choice = False, Correct_choice = False, Reward = False) = 0 /
U(Right_Choice = False, Correct_choice = True, Reward = False) =  0 /

@author: ibladmin
TODO:  Description of psy_df

"""

import statsmodels.api as sm
import seaborn as sns; sns.set()
import numpy as np
import pandas as pd
import time
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold

 
def  glm_logit(psy_df, sex_diff = True):

   ##calculate useful variables
    
    #Calculate signed contrast
    if not 'signed_contrasts' in psy_df:
        psy_df.loc[:,'contrastRight'] = psy_df['contrastRight'].fillna(0)
        psy_df.loc[:,'contrastLeft']  = psy_df['contrastLeft'].fillna(0)
        psy_df.loc[:,'signed_contrasts'] =  (psy_df['contrastRight'] - psy_df['contrastLeft'])

        
    #Add sex if not present
    if not 'sex' in psy_df:
        psy_df.loc[:,'sex'] = np.empty([psy_df.shape[0],1])
        mice  = sorted(psy_df['mouse_name'].unique())
        for mouse in mice:
            sex = input('Sex of animal ')
            psy_df.loc[ psy_df['mouse_name'] == mouse, ['sex']]  = sex
    
    #make separate datafrme 
    data =  psy_df.loc[ :, ['sex', 'mouse_name', 'feedbackType', 'signed_contrasts', 'choice','ses']]
    
    ## Build predictor matrix
    
    #Rewardeded choices: 
    data.loc[(data['choice'] == -1) & (data['feedbackType'] == -2) , 'rchoice']  = 0
    data.loc[(data['choice'] == -1) & (data['feedbackType'] == -1) , 'rchoice']  = 0
    data.loc[(data['choice'] == -1) & (data['feedbackType'] == 1) , 'rchoice']  = -1
    data.loc[(data['choice'] == 1) & (data['feedbackType'] == 1) , 'rchoice']  = 1
    data.loc[(data['choice'] == 1) & (data['feedbackType'] == -2) , 'rchoice']  = 0
    data.loc[(data['choice'] == 1) & (data['feedbackType'] == -1) , 'rchoice']  = 0
    
    #Unrewarded choices: 
    data.loc[(data['choice'] == -1) & (data['feedbackType'] == -2) , 'uchoice']  = -1
    data.loc[(data['choice'] == -1) & (data['feedbackType'] == -1) , 'uchoice']  = -1
    data.loc[(data['choice'] == -1) & (data['feedbackType'] == 1) , 'uchoice']  = 0
    data.loc[(data['choice'] == 1) & (data['feedbackType'] == 1) , 'uchoice']  = 0
    data.loc[(data['choice'] == 1) & (data['feedbackType'] == -2) , 'uchoice']  = 1
    data.loc[(data['choice'] == 1) & (data['feedbackType'] == -1) , 'uchoice']  = 1
    
    #Drop nogos
    data = data.drop(data.index[data['feedbackType'] == 0],axis=0)
    ## Change -1 for 0 in choice 
    data.loc[(data['choice'] == -1), 'choice'] = 0
    
    #make Revidence and LEvidence
    data.loc[(data['signed_contrasts'] >= 0), 'Revidence'] = data.loc[(data['signed_contrasts'] >= 0), 'signed_contrasts'].abs()
    data.loc[(data['signed_contrasts'] <= 0), 'Revidence'] = 0
    data.loc[(data['signed_contrasts'] <= 0), 'Levidence'] = data.loc[(data['signed_contrasts'] <= 0), 'signed_contrasts'].abs()
    data.loc[(data['signed_contrasts'] >= 0), 'Levidence'] = 0
    
    
    #previous choices and evidence
    
    no_tback = 5 #no of trials back
    
    start = time.time()
    for date in sorted(data['ses'].unique()):
        for i in range(no_tback):
            data.loc[data['ses'] == date,'rchoice%s' %str(i+1)] =  data.loc[data['ses'] == date,'rchoice'].shift(i+1) #no point in 0 shift
            data.loc[data['ses'] == date,'uchoice%s' %str(i+1)] =  data.loc[data['ses'] == date,'uchoice'].shift(i+1) #no point in 0 shift
            data.loc[data['ses'] == date,'Levidence%s' %str(i+1)] =  data.loc[data['ses'] == date,'Levidence'].shift(i+1) #no point in 0 shift
            data.loc[data['ses'] == date,'Revidence%s' %str(i+1)] =  data.loc[data['ses'] == date,'Revidence'].shift(i+1) #no point in 0 shift
    end = time.time()
    print(end - start)
    
    
    #Remove first 5 trials from each ses
    #for date in sorted(data['ses'].unique()):
    #    data  = data.drop(data.index[data['ses'] == date][0:5] ,axis=0)
        
    #data = data.reset_index()
    #Drop unnecessary elements
    #data =  data.drop(columns  = ['feedbackType','sex', 'ses','signed_contrasts', 'index'])
    
    data =  data.dropna()
    
    mdata = data.loc[(data['sex'] == 'M')]
    fdata = data.loc[(data['sex'] == 'F')]
    ## construct our model, with contrast as a variable
    
    ##Bayeasian mixed effects #need to change ident and exog_VC to account for mixed effects
    if sex_diff==True:
        mresult, mr2 = load_regression(mdata)
        fresult, fr2 = load_regression(fdata)
        result, r2  =  load_regression(data)
        return mresult, fresult, result, mr2, fr2, r2
    
    
    else:  
        result, r2  =  load_regression(data)
        return result,  r2
    
    

def load_regression (data, mixed_effects  = False):
    
    endog  = pd.DataFrame(data['choice'])
    exog  = data[[ 'Revidence', 'Levidence',\
            'rchoice1', 'uchoice1', 'Levidence1',\
                  'Revidence1', 'rchoice2', 'uchoice2', 'Levidence2', 'Revidence2',\
                  'rchoice3', 'uchoice3', 'Levidence3', 'Revidence3', 'rchoice4',\
                  'uchoice4', 'Levidence4', 'Revidence4', 'rchoice5', 'uchoice5',\
                  'Levidence5', 'Revidence5']]
    
    cols = list(exog.columns)
    
    #Normalising contrast
    for col in cols:
        col_zscore = col + '_zscore'
        exog[col_zscore] = (exog[col] - exog[col].mean())/exog[col].std(ddof=0)
    
    exog  =exog.drop(columns = [ 'Revidence', 'Levidence',\
            'rchoice1', 'uchoice1', 'Levidence1',\
                      'Revidence1', 'rchoice2', 'uchoice2', 'Levidence2', 'Revidence2',\
                      'rchoice3', 'uchoice3', 'Levidence3', 'Revidence3', 'rchoice4',\
                      'uchoice4', 'Levidence4', 'Revidence4', 'rchoice5', 'uchoice5',\
                      'Levidence5', 'Revidence5'] )
        
    
    ##Cross validation
    
    if mixed_effects == False :

    ##Logistic GLM wihtout mixed effects
        X_train, X_test, y_train, y_test = train_test_split(exog, np.ravel(endog), test_size=0.3)
        logit_model = sm.Logit(y_train,X_train)
        result=logit_model.fit()
        print(result.summary2())

        #crossvalidation
        
        
        cv = KFold(n_splits=10,  shuffle=False)
        logreg1 = LogisticRegression()
        r2  = cross_val_score(logreg1, exog, endog, cv=10)
        print( 'Accuracy  = ' , r2.mean())

       
    #cross validate  with sklearn 
    #NOTE: Currently cross validating without mixed effects
    
        logreg = LogisticRegression()
        logreg.fit(X_train, y_train)
        accu = logreg.score(X_test, y_test)
        print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(logreg.score(X_test, y_test)))

    return result, r2

def plot_glm(psy_df, result, r2):
    """
    INPUT:  psy_df, result of regression, r2 of regressions
    OUTPUT: Dataframe with data for plotting  + significance
    """
    
    results  =  pd.DataFrame({"Predictors": result.model.exog_names , "Coef" : result.params.values,\
                              "SEM": result.bse.values, "Significant": result.pvalues < 0.05/len(result.model.exog_names)})
    
    
    #Plotting
    fig, ax = plt.subplots(figsize=(12, 9))
    ax  = sns.barplot(x = 'Predictors', y = 'Coef', data=results, yerr= results['SEM'])    
    ax.set_xticklabels(results['Predictors'], rotation=-90)
    ax.set_ylabel('coef')
    ax.axhline(y=0, linestyle='--', color='black', linewidth=2)
    fig.suptitle ('GLM Biased Blocks')
    
    return results


    
    
