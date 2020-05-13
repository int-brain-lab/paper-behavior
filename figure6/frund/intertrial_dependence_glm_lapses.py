"""
Figure description:
GLM incorporating psychometric features such lapses
Based on code from Frund et al.2014 Journal of Vision
Code is in python2.7 and integration with datajoint is therefore difficult.
This section of the code will restructure the data so that is compatible
with the Frund et al.2014 repo: https://bitbucket.org/mackelab/serial_decision/src/master/

Author:  alejandro pan-vazquez
"""

#Import general use modules
import sys
import pandas as pd
import numpy as np
import datajoint as dj
dj.config['database.host'] = 'datajoint.internationalbrainlab.org'
from ibl_pipeline import reference, subject, action, acquisition, data, behavior
sys.path.insert(0, '/Users/alex/Documents/PYTHON/analysis/python/')
sys.path.insert(0, '/Users/alex/Documents/PYTHON/analysis/paper-behavior/')

from dj_tools import *
import os
import seaborn
import figure_style
from matplotlib.lines import Line2D

#Functions

def uuid2str(uuid):
	string  = str(uuid)
	return string

def make_target_column(behav):
	"""
	:param behav: Dataframe of behavior trials
	:return: Dataframe with target colum
	"""
	behav['target'] = np.nan
	behav.loc[(behav['signed_contrast'] > 0), 'target'] = 1
	behav.loc[(behav['signed_contrast'] < 0), 'target'] = 0
	behav.loc[(behav['signed_contrast'] == 0) & (behav['trial_feedback_type'] == 1), 'target'] = behav['choice']  # Assign correct reponse to 0 trials
	behav.loc[(behav['signed_contrast'] == 0) & (behav['trial_feedback_type'] == 0), 'target'] = behav[ 'choice'] * -1  # Assign correct reponse to 0  contrast trials
	behav.loc[(behav['target'] == -1), 'target'] = 0  # Assign correct reponse to 0 contrast trials,  turn -1 to  0
	return behav

def make_condition_column(behav):
	"""
	:param behav: Dataframe of behavior trials
	:return: Dataframe with condition colum
	"""
	behav['condition'] = np.nan
	behav.loc[(behav['probabilityLeft'] == 50), 'condition'] = 0
	behav.loc[(behav['probabilityLeft'] == 80), 'condition'] = 1
	behav.loc[(behav['probabilityLeft'] == 20), 'condition'] = 1
	return behav



def negative2zero (behav,column):
	"""
	:param behav: Dataframe of behavior trials
	:param column: Column where -1 should be 1
	:return: Dataframe without -1
	"""
	behav.loc[(behav[column] == -1), column] = 0
	return behav

def export(behav, path):
	"""
	:param behav: Dataframe of behavior trials
	:param path: Output path for txt files
	:return: txt files for each mouse in Frund et al format
	"""
	for mouse in behav['subject_uuid'].unique():
		export_dat = behav.loc[(behav['subject_uuid']== mouse), ['block' , 'condition', 'stimulus', 'target', 'choice']]
		export_dat = export_dat.dropna() #Drop trials with nans
		np.savetxt(path+str(mouse)+'.txt', export_dat, fmt = [('%1i'), ('%1i'), ('%3f'),('%1i'),('%1i')])
	return

#Set path for data
"""
You will need to modify the paths below  if running this code on your own computer
"""

path_training = '/Users/alex/Documents/Postdoc/GLM_behavior_paper/Data_by_animal/training/'
path_biased = '/Users/alex/Documents/Postdoc/GLM_behavior_paper/Data_by_animal/biased/'
path_model_analysis  =  '/Users/alex/Documents/PYTHON/trialhistory_frund/analysis.py'
path_model2array = '/Users/alex/Documents/PYTHON/analysis/paper-behavior/model2array.py'

#Import data from all the animals in both stages of training
use_subjects = subject.Subject * subject.SubjectLab * subject.SubjectProject & 'subject_project="ibl_neuropixel_brainwide_01"'
sess = (acquisition.Session & (behavior.TrialSet.Trial() & 'ABS(trial_stim_contrast_left-0)<0.0001' \
	& 'ABS(trial_stim_contrast_right-0)<0.0001') & 'task_protocol like "%trainingChoiceWorld%"') \
	* use_subjects
b 		= (behavior.TrialSet.Trial & sess) * subject.Subject() * subject.SubjectLab()
bdat 	= pd.DataFrame(b.fetch(order_by='subject_nickname, session_start_time, trial_id'))
behav 	= dj2pandas(bdat)

#Import data from biased choice world
sess = (acquisition.Session & (behavior.TrialSet.Trial() & 'ABS(trial_stim_contrast_left-0)<0.0001' \
	& 'ABS(trial_stim_contrast_right-0)<0.0001') & 'task_protocol like "%biasedChoiceWorld%"') * use_subjects
b 				= (behavior.TrialSet.Trial & sess) * subject.Subject() * subject.SubjectLab()
bdat 			= pd.DataFrame(b.fetch(order_by='subject_nickname, session_start_time, trial_id'))
behav_biased 	= dj2pandas(bdat)

#Store lab identities for later
mousenlabid = behav[['subject_uuid', 'lab_name']]
mousenlabid  = mousenlabid.drop_duplicates()
mousenlabid['subject_uuid']  = mousenlabid['subject_uuid'].apply(uuid2str) #uuid to str for easier search later
#Transform everything to Frund et al.2014
"""
File type: Txt file

Columns:  
5 -> Blocks/Conditions/Stimulus/Target/Response 
In our case these will be the following{ Blocks: no blocks at the moment,  Conditions: block_type (based on
probability left, Stimulus: signed_contrast, Target: correct_response (will be caculated below), Responce: choice}

Target and response description: 
Left contrast =  0
Right Contrast = 1
Right response  = 1
Left response  = 0 

Conditions description: 
50% left  = 0 
80% left=  1
20%  left = 1

Condition can have different slope but history weights are the same
Blocks in this case does not refer to something simular to our blocks 
Rows: Trials
One files per animal
"""
# 1 - Prepare condition column
behav['block'] = 0
behav_biased['block'] = 0

# 2 - Prepare condition column
behav = make_condition_column(behav)
behav_biased = make_condition_column(behav_biased)

# 3 -  Prepare stimulus column
behav['stimulus'] = behav['signed_contrast']/100
behav_biased['stimulus'] = behav_biased['signed_contrast']/100

# 4 - Prepare target column
behav = make_target_column(behav)
behav_biased = make_target_column(behav_biased)

# 5 - Prepare Response trials
behav =  negative2zero(behav,'choice')
behav_biased =  negative2zero(behav_biased,'choice')

#Export variables to data folder
export(behav, path_training)
export(behav_biased, path_biased)

#Model (it might take a wile)
"""
Runs model on python2.7
"""
for mouse in os.listdir(path_training):
	os.system('python2.7' + ' ' + path_model_analysis + ' ' + '-n0' + ' ' + path_training + mouse)

for mouse in os.listdir(path_biased):
	os.system('python2.7' + ' ' + path_model_analysis + ' ' + '-n0' + ' ' + path_biased + mouse)

#Extract weight and pickle for retrieving them in 3.7
"""
Concatanates Kernel Values
"""

os.system('python2.7' + ' ' + path_model2array + ' ' + path_training + ' ' + path_training)
os.system('python2.7' + ' ' + path_model2array + ' ' + path_biased + ' ' + path_biased)

#Import concatenated kernels and populate datraframe
kr_training = np.load(path_training + 'kr.npy')
kz_training = np.load(path_training + 'kz.npy')
kr_biased = np.load(path_biased + 'kr.npy')
kz_biased = np.load(path_biased + 'kz.npy')
training_names = pd.read_pickle(path_training + 'mouse_names.pkl')
biased_names = pd.read_pickle(path_biased + 'mouse_names.pkl')

regressors = pd.DataFrame(columns=['mouse_name','lab_name','training_reward_weights_1','training_reward_weights_2','training_reward_weights_3', \
							'training_reward_weights_4','training_reward_weights_5', 'training_reward_weights_6', 'training_reward_weights_7',\
							'training_stim_weights_1','training_stim_weights_2','training_stim_weights_3', \
							'training_stim_weights_4','training_stim_weights_5', 'training_stim_weights_6', 'training_stim_weights_7', \
								   'biased_reward_weights_1', 'biased_reward_weights_2', 'biased_reward_weights_3' ,\
								 'biased_reward_weights_4', 'biased_reward_weights_5', 'biased_reward_weights_6',
								'biased_reward_weights_7', \
								'biased_stim_weights_1', 'biased_stim_weights_2', 'biased_stim_weights_3', \
								'biased_stim_weights_4', 'biased_stim_weights_5', 'biased_stim_weights_6', 'biased_stim_weights_7'])

regressors['mouse_name'] = training_names
#Populate training weights
regressors[['training_reward_weights_1','training_reward_weights_2','training_reward_weights_3', \
							'training_reward_weights_4','training_reward_weights_5',\
		   'training_reward_weights_6', 'training_reward_weights_7']]  =  kr_training
regressors[['training_stim_weights_1','training_stim_weights_2','training_stim_weights_3', \
							'training_stim_weights_4','training_stim_weights_5', 'training_stim_weights_6', 'training_stim_weights_7']] = kz_training


#Populate biased weights (not all training mice might have gotten to biased choice world)

for i, mouse in enumerate(biased_names):
	regressors.loc[(regressors['mouse_name'] == mouse), ['biased_reward_weights_1', 'biased_reward_weights_2', 'biased_reward_weights_3' ,\
								 'biased_reward_weights_4', 'biased_reward_weights_5', 'biased_reward_weights_6',
								'biased_reward_weights_7']] = kr_biased[i]
	regressors.loc[(regressors['mouse_name'] == mouse), ['biased_stim_weights_1', 'biased_stim_weights_2', 'biased_stim_weights_3', \
								'biased_stim_weights_4', 'biased_stim_weights_5', 'biased_stim_weights_6',\
														 'biased_stim_weights_7']] = kz_biased[i]

#Assing a lab identity to each mouse
for mouse in regressors['mouse_name']:
	regressors.loc[regressors['mouse_name']==mouse, 'lab_name'] = mousenlabid.loc[(mousenlabid['subject_uuid'] == mouse), 'lab_name'].item()

#There must be a better method to transform weight columns into float...
regressors[['training_reward_weights_1','training_reward_weights_2','training_reward_weights_3', \
							'training_reward_weights_4','training_reward_weights_5', 'training_reward_weights_6', 'training_reward_weights_7',\
							'training_stim_weights_1','training_stim_weights_2','training_stim_weights_3', \
							'training_stim_weights_4','training_stim_weights_5', 'training_stim_weights_6', 'training_stim_weights_7', \
								   'biased_reward_weights_1', 'biased_reward_weights_2', 'biased_reward_weights_3' ,\
								 'biased_reward_weights_4', 'biased_reward_weights_5', 'biased_reward_weights_6',
								'biased_reward_weights_7', \
								'biased_stim_weights_1', 'biased_stim_weights_2', 'biased_stim_weights_3', \
								'biased_stim_weights_4', 'biased_stim_weights_5', 'biased_stim_weights_6', 'biased_stim_weights_7']] = regressors[['training_reward_weights_1','training_reward_weights_2','training_reward_weights_3', \
							'training_reward_weights_4','training_reward_weights_5', 'training_reward_weights_6', 'training_reward_weights_7',\
							'training_stim_weights_1','training_stim_weights_2','training_stim_weights_3', \
							'training_stim_weights_4','training_stim_weights_5', 'training_stim_weights_6', 'training_stim_weights_7', \
								   'biased_reward_weights_1', 'biased_reward_weights_2', 'biased_reward_weights_3' ,\
								 'biased_reward_weights_4', 'biased_reward_weights_5', 'biased_reward_weights_6',
								'biased_reward_weights_7', \
								'biased_stim_weights_1', 'biased_stim_weights_2', 'biased_stim_weights_3', \
								'biased_stim_weights_4', 'biased_stim_weights_5', 'biased_stim_weights_6', 'biased_stim_weights_7']].astype(float)

#Make a separate dataframe for separate plotting

reg_plot  = pd.melt(regressors, id_vars=['mouse_name','lab_name'], value_vars=['training_reward_weights_1','training_reward_weights_2','training_reward_weights_3', \
							'training_reward_weights_4','training_reward_weights_5', 'training_reward_weights_6', 'training_reward_weights_7',\
							'training_stim_weights_1','training_stim_weights_2','training_stim_weights_3', \
							'training_stim_weights_4','training_stim_weights_5', 'training_stim_weights_6', 'training_stim_weights_7', \
								   'biased_reward_weights_1', 'biased_reward_weights_2', 'biased_reward_weights_3' ,\
								 'biased_reward_weights_4', 'biased_reward_weights_5', 'biased_reward_weights_6',
								'biased_reward_weights_7', \
								'biased_stim_weights_1', 'biased_stim_weights_2', 'biased_stim_weights_3', \
								'biased_stim_weights_4', 'biased_stim_weights_5', 'biased_stim_weights_6', 'biased_stim_weights_7'])
reg_plot['Lag'] =np.nan
reg_plot['weight'] =np.nan

for i in reg_plot['variable'].unique():
    reg_plot.loc[(reg_plot['variable']== i),'Lag']= int(i[-1])
    reg_plot.loc[(reg_plot['variable']== i),'weight'] = i[:-10]


"""
Plot regressors as a function of lag and summed 
"""

figure_style.seaborn_style()
regression, ax =  plt.subplots(2,2,figsize=(12, 9))
sns.despine()
scatter = sns.scatterplot(x = 'Lag', y = 'value', data= reg_plot.loc[(reg_plot['weight']\
            == 'training_reward') | (reg_plot['weight']\
            == 'training_stim')],  hue = 'weight', style = 'lab_name', ax = ax[0,0])
sns.lineplot(x = 'Lag', y = 'value', data= reg_plot.loc[(reg_plot['weight']\
            == 'training_reward') | (reg_plot['weight']\
            == 'training_stim')], hue = 'weight', ax = ax[0,0], color = ['red','blue'])
sns.scatterplot(x = 'Lag', y = 'value', data= reg_plot.loc[(reg_plot['weight']\
            == 'biased_reward') | (reg_plot['weight']\
            == 'biased_stim')],  hue = 'weight', style = 'lab_name', ax = ax[0,1])
ax[0,1].get_legend().remove()
sns.lineplot(x = 'Lag', y = 'value', data= reg_plot.loc[(reg_plot['weight']\
            == 'biased_reward') | (reg_plot['weight']\
            == 'biased_stim')], hue = 'weight', ax = ax[0,1])
sns.scatterplot(x = 'training_reward_weights_1', y =  'training_stim_weights_1', data =  weight_scat, ax  = ax[1,0], style = 'lab_name', color='black')
sns.scatterplot(x = 'biased_reward_weights_1', y =  'biased_stim_weights_1', data =  weight_scat, ax  = ax[1,1], style = 'lab_name',color='black')
ax[1,1].set_xlim([-1.5,1.5])
ax[1,1].set_ylim([-1.5,1.5])
ax[1,1].get_legend().remove()
ax[0,0].get_legend().remove()
ax[0,0].set_title('Level 1')
ax[0,1].set_title('Level 2')
ax[0,1].set_ylabel('Weight')
ax[0,0].set_ylabel('Weight')
ax[1,0].set_ylabel('Stimulus Weight (1 trial back)')
ax[1,1].set_ylabel('Stimulus Weight (1 trial back)')
ax[0,1].set_xlabel('Trials back')
ax[0,0].set_xlabel('Trials back')
ax[1,0].set_xlabel('Response Weight (1 trial back)')
ax[1,1].set_xlabel('Response Weight (1 trial back)')
ax[0,0].set_ylim([-1.5,1.5])
ax[0,1].set_ylim([-1.5,1.5])
ax[0,1].get_legend().remove()
ax[1,0].set_xlim([-1.5,1.5])
ax[1,0].set_ylim([-1.5,1.5])
ax[1,0].get_legend().remove()
ax[1,0].plot(ax[1,0].get_xlim(), ax[1,0].get_ylim(), ls="--", c=".3")
ax[1,0].plot(np.array(ax[1,0].get_xlim()),np.array(ax[1,0].get_ylim())*-1, ls="--", c=".3")
ax[1,1].plot(ax[1,1].get_xlim(), ax[1,1].get_ylim(), ls="--", c=".3")
ax[1,1].plot(np.array(ax[1,1].get_xlim()),np.array(ax[1,1].get_ylim())*-1, ls="--", c=".3")
ax[1,0].set_xticks([-1.5, 0, 1.5])
ax[1,0].set_yticks([-1.5, 0, 1.5])
ax[1,1].set_xticks([-1.5, 0, 1.5])
ax[1,1].set_yticks([-1.5, 0, 1.5])
ax[1,1].text(0,1.3, 'win stay - lose switch',horizontalalignment='center', verticalalignment='center')
ax[1,1].text(0,-1.3, 'win switch - lose stay',horizontalalignment='center', verticalalignment='center')
ax[1,1].text(-1.3,0, 'switch',horizontalalignment='center', verticalalignment='center', rotation= 90)
ax[1,1].text(1.3,0, 'stay',horizontalalignment='center', verticalalignment='center', rotation= 270)
ax[1,0].text(0,1.3, 'win stay - lose switch',horizontalalignment='center', verticalalignment='center')
ax[1,0].text(0,-1.3, 'win switch - lose stay',horizontalalignment='center', verticalalignment='center')
ax[1,0].text(-1.3,0, 'switch',horizontalalignment='center', verticalalignment='center', rotation= 90)
ax[1,0].text(1.3,0, 'stay',horizontalalignment='center', verticalalignment='center', rotation= 270)
ax[0,0].legend(('Response','Stimulus'))
ax[0,1].legend(('Response','Stimulus'))

regression.savefig("regressors.pdf")
regression.savefig("regressors.svg")


"""
Optional: Comparison training versus biased response kernels
sns.set()
regression, ax =  plt.subplots(figsize=(12, 9))
sns.swarmplot(x = 'weight', y = 'value', data= reg_plot.loc[(reg_plot['variable']\
            == 'training_reward_weights_1') | (reg_plot['variable']\
            == 'biased_reward_weights_1')],  hue = 'lab_name', alpha = 0.5)
sns.barplot(x = 'weight', y = 'value', data= reg_plot.loc[(reg_plot['variable']\
            == 'training_reward_weights_1') | (reg_plot['variable']\
            == 'biased_reward_weights_1')])
"""  