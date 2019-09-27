#requires Alex glm module


import matplotlib.pyplot as plt
import pandas as pd
## CONNECT TO datajoint
import datajoint as dj
dj.config['database.host'] = 'datajoint.internationalbrainlab.org'
from ibl_pipeline.analyses import behavior as behavior_analysis
from ibl_pipeline import reference, subject, behavior
from alexfigs_datajoint_functions import *  # this has all plotting functions
import seaborn as sns
from glm import *
from paper_behavior_functions import figpath
import os  # For making paths

# Set save path for figures
save_path = figpath()

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
#Select only biased blocks
psy_df =  trials_ibl.loc[(trials_ibl['trial_stim_prob_left'] == 0.8) | (trials_ibl['trial_stim_prob_left'] == 0.2)]

pool_rew_predictors  = pd.DataFrame()
pool_urew_predictors  = pd.DataFrame()



for mouse in psy_df['subject_nickname'].unique():
    mouse_result, mouse_r2  = glm_logit(psy_df.loc[(psy_df['subject_nickname']==mouse)], sex_diff = False)
    mouse_results  =  pd.DataFrame({"Predictors": mouse_result.model.exog_names , "Coef" : mouse_result.params.values,\
                          "SEM": mouse_result.bse.values, "Sex": "M"})
    mouse_results.set_index('Predictors', inplace=True)
    mouse_rew_predictors =  mouse_results.loc[['rchoice1_zscore','rchoice2_zscore', 'rchoice3_zscore','rchoice4_zscore','rchoice5_zscore'],['Coef']].reset_index()
    mouse_rew_predictors['subject_nickname'] = mouse
    mouse_urew_predictors =  mouse_results.loc[['uchoice1_zscore','uchoice2_zscore', 'uchoice3_zscore','uchoice4_zscore','uchoice5_zscore'],['Coef']].reset_index()
    mouse_urew_predictors['subject_nickname'] = mouse
    pool_rew_predictors = pool_rew_predictors.append(mouse_rew_predictors)
    pool_urew_predictors = pool_urew_predictors.append(mouse_urew_predictors)


pool_rew_predictors = pool_rew_predictors.reset_index()
pool_rew_predictors.loc[:,'index']  = pool_rew_predictors.loc[:,'index'] + 1
pool_urew_predictors = pool_urew_predictors.reset_index()
pool_urew_predictors.loc[:,'index']  = pool_urew_predictors.loc[:,'index'] + 1


sns.set()

regression, ax =  plt.subplots(figsize=(12, 9))
ax  = sns.scatterplot(x = 'index', y = 'Coef', data=pool_rew_predictors, alpha=0.2)    
ax  = sns.scatterplot(x = 'index', y = 'Coef', data=pool_urew_predictors, alpha=0.2,  color ='red')    
ax  = sns.lineplot(x = 'index', y = 'Coef', data=pool_rew_predictors)    
ax  = sns.lineplot(x = 'index', y = 'Coef', data=pool_urew_predictors, color='red')    
ax.set_xticks([1,2,3,4,5])
ax.set_xlabel('Trials back')
ax.legend(('Rewarded','Unrewarded'))
regression.suptitle ('Regressors')
regression.savefig(os.path.join(save_path, "regressors.pdf"))
