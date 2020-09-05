"""
Choice variability across contrasts within each lab and across labs

@author: Anne Urai
2 September 2020
"""

import numpy as np
import seaborn as sns
import os
from os.path import join
import pandas as pd
import matplotlib.pyplot as plt
from paper_behavior_functions import (figpath, seaborn_style, group_colors,
                                      query_sessions_around_criterion, institution_map,
                                      FIGURE_HEIGHT, FIGURE_WIDTH, QUERY,
                                      dj2pandas)
# import wrappers etc
from ibl_pipeline import reference, subject, behavior
from sklearn.utils import shuffle

# progress bar
from tqdm.auto import tqdm
tqdm.pandas(desc="computing")

# Initialize
seaborn_style()
figpath = figpath()
pal = group_colors()
institution_map, col_names = institution_map()
col_names = col_names[:-1]

# ================================= #
# GET DATA FROM TRAINED ANIMALS
# ================================= #

if QUERY is True:
    # query sessions
    use_sessions, use_days = query_sessions_around_criterion(criterion='trained',
                                                             days_from_criterion=[2,0],
                                                             as_dataframe=False,
                                                             force_cutoff=True)
    # Trial data to fetch
    trial_fields = ('trial_stim_contrast_left',
                    'trial_stim_contrast_right',
                    'trial_response_time',
                    'trial_stim_prob_left',
                    'trial_feedback_type',
                    'trial_stim_on_time',
                    'trial_response_choice')

    # Query trial data for sessions and subject name and lab info
    trials = use_sessions.proj('task_protocol') * behavior.TrialSet.Trial.proj(*trial_fields)
    subject_info = subject.Subject.proj('subject_nickname') * \
        (subject.SubjectLab * reference.Lab).proj('institution_short')

    # Fetch, join and sort data as a pandas DataFrame
    behav = dj2pandas(trials.fetch(format='frame')
                      .join(subject_info.fetch(format='frame'))
                      .sort_values(by=['institution_short', 'subject_nickname',
                                       'session_start_time', 'trial_id'])
                      .reset_index())
    behav['institution_code'] = behav.institution_short.map(institution_map)
else:
    behav = pd.read_csv(join('data', 'Fig3'))

# COMPUTE AVERAGE CHOICE BEHAVIOR FOR EACH SJ
df = behav.groupby(['institution_code', 'subject_nickname', 'signed_contrast']).agg(
    {'choice2': 'mean'}).reset_index()
df.drop(df[ df['signed_contrast'].isin([50., -50])].index, inplace=True)


# DEFINE ACROSS-MOUSE CHOICE VARIABILITY
def choice_variability(df):

    # FIRST, GROUP BY CONTRAST AND MOUSE
    choicedat = df.pivot_table(values='choice2', columns='subject_nickname', index='signed_contrast')

    # COMPUTE THE ACROSS-MOUSE VARIANCE FOR EACH CONTRAST
    # AVERAGE VARIANCE ACROSS CONTRASTS
    choice_var = choicedat.var(axis=1).mean()

    return choice_var

# ================================================================== #
# COMPUTE CHOICE VARIABILITY FOR EACH LAB
# ================================================================== #

choice_variability_perlab = df.groupby(['institution_code']).progress_apply(choice_variability).reset_index()
choice_variability_perlab['x'] = 0
choice_variability_perlab['choice_var'] = choice_variability_perlab[0]

# ================================================================== #
# SAME, BUT ON SHUFFLED DATA
# ================================================================== #

nshuf = 1000
choice_variability_shuffled = []

for s in tqdm(range(nshuf)):
    # use scikit learn to shuffle lab labels
    df['new_lab'] = shuffle(df['institution_code']).reset_index()['institution_code']
    # assert (all(behav.groupby(['institution_code'])['new_lab'].nunique() > 1))
    choice_variability_shuffled.append(df.groupby(['new_lab']).apply(choice_variability).mean())

# ================================================================== #
# PLOT
# ================================================================== #

# Plot
f, ax1 = plt.subplots(1, 1, figsize=(FIGURE_WIDTH / 5, FIGURE_HEIGHT))
# first, data
sns.swarmplot(data = choice_variability_perlab,
              x ='x', y='choice_var', hue_order=col_names,
              hue = 'institution_code',
              palette = pal, marker='.', ax=ax1, zorder=0)
ax1.plot(0, choice_variability_perlab['choice_var'].mean(),
             color='black', linewidth=0, marker='_', markersize=13)
ax1.get_legend().set_visible(False)
# then, shuffled distribution next to it
sns.violinplot(x=np.concatenate((np.zeros(nshuf), np.ones(nshuf))),
              y=np.concatenate((np.empty((nshuf))*np.nan, choice_variability_shuffled)),
               palette=[[1,1,1]], ax=ax1)
ax1.set(ylabel='Choice variability\nacross mice', xlabel='', yticks=[0, 0.01, 0.02, 0.03])
ax1.set_xticklabels(['Data', 'Shuffle'], ha='center')
plt.tight_layout()
sns.despine(trim=True)
f.savefig(os.path.join(figpath, "across_mouse_var.pdf"))

# WHAT IS THE P-VALUE COMPARED TO THE  NULL DISTRIBUTION?
pval = np.min([np.mean(choice_variability_shuffled > choice_variability_perlab['choice_var'].mean()),
               np.mean(choice_variability_shuffled < choice_variability_perlab['choice_var'].mean())])
print('p-value = %.4f'%pval)