

import time, re, datetime, os, glob
from datetime import timedelta
import seaborn as sns

import matplotlib as mpl
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
from IPython import embed as shell
from scipy import stats

## CONNECT TO datajoint

import datajoint as dj
dj.config['database.host'] = 'datajoint.internationalbrainlab.org'

import datajoint as dj
from ibl_pipeline.analyses import behavior as behavior_analysis
from ibl_pipeline import reference, subject, action, acquisition, data, behavior
from ibl_pipeline.analyses import behavior as behavior_analysis
from alexfigs_datajoint_functions import *  # this has all plotting functions


#Collect all alyx data

ephys_mice = pd.DataFrame(((subject.Subject()  & 'sex!="U"')  * \
       (behavior_analysis.SessionTrainingStatus() & 'training_status="ready for ephys"')).fetch('subject_nickname', as_dict=True))[0].unique()

pars_05_pool  = pd.DataFrame()
pars_02_pool  = pd.DataFrame()
pars_08_pool  = pd.DataFrame()

#Pool psychometric data
for mouse in ephys_mice:
    pars = pd.DataFrame((behavior_analysis.BehavioralSummaryByDate.PsychResults * \
                                         subject.Subject * subject.SubjectLab & 'subject_nickname="%s"'%mouse).fetch(as_dict=True))
    
    pars_05  = pars.loc[(pars['prob_left'] == 0.5)].reset_index()
    pars_05 =  pars_05.drop_duplicates('session_date').reset_index()
    
    pars_02  = pars.loc[(pars['prob_left'] == 0.2)].reset_index()
    pars_02 =  pars_02.drop_duplicates('session_date').reset_index()
    
    pars_08  = pars.loc[(pars['prob_left'] == 0.8)].reset_index()
    pars_08 =  pars_08.drop_duplicates('session_date').reset_index()
    
    pars_05_pool = pars_05_pool.append(pars_05.loc[:,['level_0','threshold','lapse_high','lapse_low','bias','lab_name','subject_nickname']],sort=False)
    pars_02_pool = pars_02_pool.append(pars_02.loc[:,['level_0','threshold','lapse_high','lapse_low','bias','lab_name','subject_nickname']],sort=False)
    pars_08_pool = pars_08_pool.append(pars_08.loc[:,['level_0','threshold','lapse_high','lapse_low','bias','lab_name','subject_nickname']],sort=False)

#Add n numbers
pars_05_pool ['n']= np.nan
pars_02_pool ['n']= np.nan
pars_08_pool ['n']= np.nan

for lab in pars_05_pool['lab_name'].unique():
    pars_05_pool['n'].loc[(pars_05_pool['lab_name']== lab)] = pars_05_pool.loc[(pars_05_pool['lab_name']==lab)].subject_nickname.nunique()
    pars_02_pool['n'].loc[(pars_02_pool['lab_name']== lab)] = pars_02_pool.loc[(pars_02_pool['lab_name']==lab)].subject_nickname.nunique()
    pars_08_pool['n'].loc[(pars_08_pool['lab_name']== lab)] = pars_08_pool.loc[(pars_08_pool['lab_name']==lab)].subject_nickname.nunique()
   

##Plot
sns.set()
psychometric_measures, ax = plt.subplots(2,3,figsize=(20,12))
#Threshold
sns.lineplot(x="level_0", y="threshold", data=pars_05_pool, hue="lab_name", ax=ax[0,0], lw=0.8, \
             alpha =0.5, err_kws={'alpha' :0.1})
sns.lineplot(x="level_0", y="threshold", data = pars_05_pool, ax=ax[0,0],  color='black', lw=2)
ax[0,0].legend_.remove()
ax[0,0].set_xlim([0, 50])
ax[0,0].set_xlabel('Session Number')
ax[0,0].set_ylabel('Threshold')

#Lapse Right
sns.lineplot(x="level_0", y="lapse_high", data=pars_05_pool,hue="lab_name",  ax=ax[0,1], lw=0.8, \
             alpha =0.5, err_kws={'alpha' :0.1})
sns.lineplot(x="level_0", y="lapse_high", data = pars_05_pool, ax=ax[0,1],  color='black', lw=2)
ax[0,1].legend_.remove()
ax[0,1].set_xlim([0, 50])
ax[0,1].set_xlabel('Session Number')
ax[0,1].set_ylabel('Lapse Right')

#Lapse Left
sns.lineplot(x="level_0", y="lapse_low", data=pars_05_pool,hue="lab_name",  ax=ax[0,2], lw=0.8, \
             alpha =0.5, err_kws={'alpha' :0.1})
sns.lineplot(x="level_0", y="lapse_low", data = pars_05_pool, ax=ax[0,2],  color='black', lw=2)
ax[0,2].legend_.remove()
ax[0,2].set_xlim([0, 50])
ax[0,2].set_xlabel('Session Number')
ax[0,2].set_ylabel('Lapse Left')

#Bias  20% Right block
sns.lineplot(x="level_0", y="bias", data=pars_02_pool,hue="lab_name", ax=ax[1,0], lw=0.8, \
             alpha =0.5, err_kws={'alpha' :0.1})
sns.lineplot(x="level_0", y="bias", data = pars_02_pool, ax=ax[1,0],  color='black', lw=2)
ax[1,0].legend_.remove()
ax[1,0].set_xlim([0, 30])
ax[1,0].set_ylim([-50, 50])
ax[1,0].set_xlabel('Session Number')
ax[1,0].set_ylabel('Bias  20% Right block')

#Bias  80% Right block
sns.lineplot(x="level_0", y="bias", data=pars_08_pool,hue="lab_name", ax=ax[1,1], lw=0.8, \
             alpha =0.5, err_kws={'alpha' :0.1})
sns.lineplot(x="level_0", y="bias", data = pars_08_pool, ax=ax[1,1],  color='black', lw=2)
ax[1,1].legend_.remove()
ax[1,1].set_xlim([0, 30])
ax[1,0].set_ylim([-50, 50])
ax[1,1].set_xlabel('Session Number')
ax[1,1].set_ylabel('Bias  80% Right block')

psychometric_measures.suptitle ('Stability of Psychometrics')
psychometric_measures.savefig("psy_stability.pdf")
