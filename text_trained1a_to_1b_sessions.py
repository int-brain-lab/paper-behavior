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
            & 'training_status = "trained_1a" OR training_status = "trained_1b"')
           .proj('subject_nickname', 'n_trials_stim', 'institution_short', 'training_status')
           .fetch(format='frame')
           .reset_index())
    ses['n_trials'] = [sum(i) for i in ses['n_trials_stim']]
else:
    ses = pd.read_csv(join(datapath(), 'Fig2c.csv'))
    use_subjects = ses['subject_uuid'].unique()  # For counting the number of subjects

ses = ses.sort_values(by=['subject_uuid', 'session_start_time'])
uni_sub = np.unique(ses['subject_uuid'])

training_time = pd.DataFrame(columns=['sessions'])
# Loop over subjects
for i_sub in range(0, len(uni_sub)):
    subj = uni_sub[i_sub]

    # Construct dataframe
    df = ses.loc[ses['subject_uuid'] == subj]
    if len(np.unique(df['training_status'])) == 2:  # Append

        # Check that the session start date is different for when reaching 1a/1b
        df = df.sort_values(by=['session_start_time'])  # Ensure data is sorted by date

        # Find index of relevant session
        indx_a = np.where(df['training_status'] == 'trained_1a')[0]
        n_row_a = indx_a[-1]  # last session with trained 1a
        indx_b = np.where(df['training_status'] == 'trained_1b')[0]
        n_row_b = indx_b[0]  # first session with trained 1b
        if n_row_a+1 != n_row_b:
            print("ERROR")
        #  Get and compare dates
        date_a = df.iloc[[n_row_a]]['session_start_time'].values
        date_a = date_a.astype('datetime64[D]')
        date_b = df.iloc[[n_row_b]]['session_start_time'].values
        date_b = date_b.astype('datetime64[D]')
        if date_a != date_b and date_b > DATE_IMPL:
            # Print for debugging purposes
            # print(f'trained_1b: {date_b}, subject uuid: {subj}')
            # Aggregate and append
            training_time_ab = pd.DataFrame(columns=['sessions'],
                                            data=df.groupby(['training_status']).size())
            training_time = training_time.append(training_time_ab.loc['trained_1a'])  # Take N session done under 1a

# Training time as a whole (N session in trained_1a before reaching trained_1b)
m_train = training_time['sessions'].mean()
s_train = training_time['sessions'].std()
slowest = training_time['sessions'].max()
fastest = training_time['sessions'].min()

n_mice = len(training_time)
print(f'using impl. date: {DATE_IMPL} : {n_mice} mice, n session from 1a>1b: {round(m_train, 2)} ± {round(s_train, 2)}')
