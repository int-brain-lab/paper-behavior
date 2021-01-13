#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
N mice with trained 1b after implementation date

@author: Gaelle Chapuis
Jan 2021
"""
from os.path import join

import pandas as pd
import numpy as np
from datetime import datetime

from paper_behavior_functions import (query_subjects, datapath, QUERY)
from ibl_pipeline.analyses import behavior as behavior_analysis

# Date at which trained_1b was implemented in DJ pipeline
DATE_IMPL = datetime.strptime('12-09-2019', '%d-%m-%Y').date()

# Query data
if QUERY is True:
    # Query sessions
    use_subjects = query_subjects()
    ses = ((use_subjects * behavior_analysis.SessionTrainingStatus * behavior_analysis.PsychResults
            & 'training_status = "trained_1b"')
           .proj('subject_nickname', 'n_trials_stim', 'institution_short', 'training_status')
           .fetch(format='frame')
           .reset_index())
    ses['n_trials'] = [sum(i) for i in ses['n_trials_stim']]
else:
    ses = pd.read_csv(join(datapath(), 'Fig2c.csv'))
    use_subjects = ses['subject_uuid'].unique()  # For counting the number of subjects

ses = ses.sort_values(by=['subject_uuid', 'session_start_time'])
uni_sub = np.unique(ses['subject_uuid'])

n_mice = 0
# Loop over subjects
for i_sub in range(0, len(uni_sub)):
    subj = uni_sub[i_sub]

    # Construct dataframe
    df = ses.loc[ses['subject_uuid'] == subj]

    # Check that the session start date is after impl. date
    df = df.sort_values(by=['session_start_time'])  # Ensure data is sorted by date

    #  Get and compare dates
    date_b = df.iloc[[0]]['session_start_time'].values
    date_b = date_b.astype('datetime64[D]')
    if date_b > DATE_IMPL:
        # Print for debugging purposes
        # print(f'trained_1b: {date_b}, subject uuid: {subj}')
        # Count
        n_mice = n_mice+1

print(f'using impl. date: {DATE_IMPL} : {n_mice} mice')
