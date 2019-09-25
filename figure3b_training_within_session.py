"""
Disengagement criteria
Anne Urai, Miles Wells, 2019
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from paper_behavior_functions import *
import datajoint as dj

endcriteria = dj.create_virtual_module('SessionEndCriteria', 'group_shared_end_criteria')
# from group_shared_end_criteria import SessionEndCriteria  # TODO load from group_shared
from ibl_pipeline import behavior, subject, acquisition

# ================================= #
# QUERY AN EXAMPLE MOUSE
# ================================= #

mouse = 'CSHL_015'
lab = 'churchlandlab'

subj = subject.Subject & 'subject_nickname = "{}"'.format(mouse)
sessions = behavior.TrialSet & subj & (acquisition.Session() - 'task_protocol LIKE "%habituation%"')

# ================================================================== #
# QUERY SESSIONS DETAILS FOR EXAMPLE 3 DAYS
# ================================================================== #

days = [3, 10, 27]
#sessions = sessions.proj(n_days='DATEDIFF(session_start_time, MIN(session_start_time))')
end_criterion = sessions * endcriteria.SessionEndCriteria
print(end_criterion)
