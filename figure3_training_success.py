#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 29 12:00:42 2019

@author: guido
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from os.path import join
import seaborn as sns
from figure_style import seaborn_style
from subject_query import query_subjects
import datajoint as dj
from ibl_pipeline import subject, acquisition, action, behavior, reference
from ibl_pipeline.analyses import behavior as behavior_analysis

# Query list of subjects
subjects = query_subjects()

trained_df = pd.DataFrame(columns=['mouse', 'lab', 'trained'])
for i, nickname in enumerate(subjects['subject_nickname']):
    if np.mod(i+1, 10) == 0:
        print('Loading data of subject %d of %d' % (i+1, len(subjects)))

    training = (behavior_analysis.SessionTrainingStatus * subject.Subject
                & 'subject_nickname="%s"' % nickname).fetch('training_status')
    if len(training) == 0 or np.sum(training == 'trained') == 0:
        trained_df.loc[i, 'trained'] = False
    else:
        trained_df.loc[i, 'trained'] = True
    trained_df.loc[i, 'mouse'] = nickname
    trained_df.loc[i, 'lab'] = subjects.loc[subjects['subject_nickname'] == nickname,
                                            'lab_name'][0]
