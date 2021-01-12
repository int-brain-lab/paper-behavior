#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Time from trained 1a > 1b

@author: Gaelle Chapuis
Jan 2021
"""
from os.path import join

import pandas as pd
import numpy as np

from paper_behavior_functions import (query_subjects, datapath, QUERY)
from ibl_pipeline.analyses import behavior as behavior_analysis

# Settings

if QUERY is True:
    # Query sessions
    use_subjects = query_subjects()
    ses = ((use_subjects * behavior_analysis.SessionTrainingStatus * behavior_analysis.PsychResults
            & 'training_status = "trained_1a" OR training_status = "trained_1b"')
           .proj('subject_nickname', 'n_trials_stim', 'institution_short', 'training_status')
           .fetch(format='frame')
           .reset_index())
    ses['n_trials'] = [sum(i) for i in ses['n_trials_stim']]
else:
    ses = pd.read_csv(join(datapath(), 'Fig2c.csv'))
    use_subjects = ses['subject_uuid'].unique()  # For counting the number of subjects

ses = ses.sort_values(by=['subject_nickname', 'training_status'])
uni_sub = np.unique(ses['subject_nickname'])

# Construct dataframe
training_time_ab = pd.DataFrame(columns=['sessions'], data=ses.groupby(['subject_nickname','training_status']).size())
training_time_b = pd.DataFrame(columns=['sessions'])
for i_sub in range(0, len(uni_sub)):
    subj = uni_sub[i_sub]
    ab = training_time_ab.loc[subj]
    if len(ab)==2:
        tr_b = ab.loc['trained_1b']
        training_time_b = training_time_b.append(tr_b)

# Training time as a whole
m_train = training_time_b['sessions'].mean()
s_train = training_time_b['sessions'].std()
slowest = training_time_b['sessions'].max()
fastest = training_time_b['sessions'].min()
