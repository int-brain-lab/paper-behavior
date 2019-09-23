"""
AVERAGE WEIGHT CURVE ON WATER RESTRICTION
Miles Wells, UCL, 2019
"""

import datajoint as dj
from ibl_pipeline import action
import matplotlib.pyplot as plt
import seaborn as sns
from paper_behavior_functions import query_subjects, seaborn_style

# All water restrictions
water_restrictions = action.WaterRestriction()

# Subjects to use for paper
subjects = query_subjects()

# Get the first water restriction for each mouse with .aggr()
subject_first_restriction = subjects.aggr(water_restrictions,
                                          restriction_start_time='min(restriction_start_time)')

# Number of days from start of water restriction
weight_days = ((action.Weighing() * subject_first_restriction)
               .proj(n_days='DATEDIFF(weighing_time, restriction_start_time)'))

# Normalize by reference weight and isolate first 30 days
norm_weight = ((action.Weighing() * water_restrictions * (weight_days & 'n_days < 30'))
               .proj(norm='weight / reference_weight'))

# dj.U('n_days') means all possible values of n_days by join or restrict with other tables,
# returns a query with n_days as primary key.
data = (dj.U('n_days')
        .aggr(weight_days * norm_weight, mean_weight='avg(norm)', std_weight='std(norm)'))

#  Fetch the data for plotting, ignoring -ve days
df = (data & 'n_days >= 0').fetch(format="frame")

# plot
plt.errorbar(df.index, df['mean_weight'], yerr=df['std_weight'])
plt.gca().set(xlabel='# Days water restricted', ylabel='Normalized weight change', ylim=[.7, 1])

#  Make fancy
seaborn_style()
sns.set()
plt.show()
