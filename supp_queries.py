"""
This file contains some supplementary queries and statistics for the paper

@author: Miles Wells
"""
from uuid import UUID

import pandas as pd
import numpy as np
import datajoint as dj
from ibl_pipeline import behavior, acquisition, subject, action
from ibl_pipeline.analyses import behavior as behavior_analysis

from paper_behavior_functions import (
    query_subjects, EXAMPLE_MOUSE, institution_map, CUTOFF_DATE, load_csv
)


############################
# Days to full proficiency #
############################
"""This section prints the mean number of days and trials required to reach full task
proficiency"""
use_subjects = query_subjects(criterion='ephys')
query = (behavior_analysis.BehavioralSummaryByDate * use_subjects)
query = subject.Subject.proj().aggr(
    query & 'session_date <= date_trained',
    'institution_short',
    n_training_days='MAX(training_day)',
    total_trials='SUM(n_trials_date)',
    n_training_weeks='MAX(training_week)'
)

df = query.fetch(order_by='institution_short', format='frame')
# Ensure correct data types
institution_map, _ = institution_map()
df['institution_code'] = df.institution_short.map(institution_map)
int_fields = ['n_training_weeks', 'n_training_days', 'total_trials']
df[int_fields] = df[int_fields].apply(pd.to_numeric)

# Fetch UUID of example mouse
example_mouse = (subject.Subject & f'subject_nickname = "{EXAMPLE_MOUSE}"').fetch1('subject_uuid')

# Training time as a whole
m_days = df['n_training_days'].mean()
s_days = df['n_training_days'].std()
slowest_days = df['n_training_days'].max()
fastest_days = df['n_training_days'].min()

m_trials = df['total_trials'].mean()
s_trials = df['total_trials'].std()
slowest_trials = df['total_trials'].max()
fastest_trials = df['total_trials'].min()

example_training_days = df.loc[example_mouse].n_training_days
example_training_trials = df.loc[example_mouse].total_trials

# Print information used in the paper
print(
    'For mice that reached full task proficiency, the average training took '
    f'{m_days:.1f} ± {s_days:.1f} days, or '
    f'{m_trials:.0f} ± {s_trials:.0f} trials (s.d., n = {len(df)}). '
    f'The example mouse from Lab 1 (Figure 2a, black) took {example_training_days}. '
    f'The fastest learner achieved proficiency in {fastest_days} days ({fastest_trials} trials), '
    f'the slowest {slowest_days} days ({slowest_trials} trials)'
)

print('Mean training times by lab:\n')
print(df.groupby('institution_code').mean())


##################################
# Session end criteria adherence #
##################################
"""Count the proportion of sessions that have an end criterion (i.e. were not ended early) and 
for these sessions, count the number of trials completed after this criterion was met."""

all_mice = query_subjects(criterion=None)  # Mice that started the training task protocol
endcriteria = dj.create_virtual_module('SessionEndCriteriaImplemented',
                                       'group_shared_end_criteria')
# Every session from mice that entered the pipeline
all_sessions = (
    (
        behavior.TrialSet
        * acquisition.Session
        * all_mice.proj('institution_short')
        & behavior.CompleteTrialSession
    ).proj(
        'n_trials', 'institution_short',
        session_date='DATE(session_start_time)',
        duration='trials_end_time - trials_start_time')
)
df1 = all_sessions.fetch(format='frame')

# Intersect these sessions with the end criteria information
fields = ('n_trials', 'institution_short', 'session_date', 'duration', 'end_status')
met_end_crit = ((all_sessions * endcriteria.SessionEndCriteriaImplemented)
                .proj('end_status', n_over='n_trials - end_status_index'))
df2 = met_end_crit.fetch(format='frame')  # All sessions with a criterion

# Join the two tables, giving all sessions with None for criteria columns where no criterion was
# met
df = df1.join(df2)
df['institution_code'] = df.institution_short.map(institution_map)
df['n_over'] = df['n_over'].astype(float)  # Cast the calculated fields to their correct dtype
df['duration'] = df['duration'].astype(float)

# Due to crashes and other impediments there may be multiple sessions in one day, with only one
# having an end criterion.  To account for this we will count the days where at least one
# session had an end criterion met for each mouse.
df['has_end_status'] = ~df['end_status'].isna()
has_end = df.groupby(by=['subject_uuid', 'session_date']).any()['has_end_status']
# Finally we should account for the days where there were multiple sessions and no end criterion,
# but the total training time for the day exceeded 90 minutes (itself one of end criteria)
total_gt_90 = df.groupby(by=['subject_uuid', 'session_date']).sum()['duration'] >= 60*90
end_or_gt = np.logical_or(has_end.values, total_gt_90.values)

print(f'Of the {len(end_or_gt)} training days, {end_or_gt.sum()} met an end criterion* '
      f'({end_or_gt.sum() / len(end_or_gt):.2%}) ' +
      f'where a training day is a day with one or more sessions for a given mouse. '
      '*One or more sessions in that day had an end status, or the sum of session durations '
      'exceeded 90 minutes; without merging sessions by day, this value is '
      f'{df["has_end_status"].sum() / len(df):.2%}.')


# Check the number of trials the occured after a session end criterion was met.
def mad(x):
    m = np.nanmedian(x)
    x = abs(x - m)
    return np.nanmedian(x)


print(f'Of the sessions with a session end status, the median number of trials over was '
      f'{df.n_over.median():.0f}, m.a.d. = {mad(df.n_over.values):.0f}, '
      f'{sum(df.has_end_status)} sessions')

print('Summary data by lab:\n')
print(
    (df
        .reset_index()
        .groupby('institution_code')
        .n_over
        .describe()
        .drop('min', axis=1)  # min trials over = 1 for all labs
    )
)

##########################################
# Number of subjects before 1b criterion #
##########################################
"""
Count the number of mice that reached biased criterion before 1b was introduced in September 2019.
"""
# Date at which trained_1b was implemented in DJ pipeline
DATE_IMPL = '2019-09-12'
n_trained_before = len(query_subjects(criterion='ephys') & f'date_trained < "{DATE_IMPL}"')
n_trained = len(query_subjects(criterion='trained'))

print('The second set, called "1b", was introduced shortly afterwards (September 2019) and '
      f'applied to {n_trained - n_trained_before} of the {n_trained} mice.')


###########################################
#   Breakdown of mice that didn't learn   #
###########################################
"""
Count the number of mice that didn't learn and summarize the reasons for them not finishing the 
pipeline.
"""
all_mice = query_subjects(criterion=None)  # Mice that started the training task protocol
still_training = all_mice * subject.Subject.aggr(behavior_analysis.SessionTrainingStatus,
                                                 session_start_time='max(session_start_time)') \
                   * behavior_analysis.SessionTrainingStatus - subject.Death \
                   & 'training_status = "in_training"' & 'session_start_time > "%s"' % CUTOFF_DATE
mice_started_training = (all_mice & (acquisition.Session() & 'task_protocol LIKE "%training%"'))
trained = query_subjects(criterion='trained')
still_training_uuids = [{'subject_uuid': id} for id in still_training.fetch('subject_uuid')]
not_trained = mice_started_training - still_training_uuids - trained.proj()

print(f'Number of mice that went into training: {len(mice_started_training)}')
print(f'Number of mice that are still in training (exclude from 1 and 2): {len(still_training)}')
print(f'Number of mice that reached trained:{len(trained)}')
print(f'Number of mice that didn\'t reach trained: {len(not_trained)}')

# Fetch list of mice that were not trained, and include the status of their final session
status = (behavior_analysis.SessionTrainingStatus * behavior_analysis.PsychResults) & behavior.CompleteTrialSession
fields = ('training_status', 'threshold', 'bias', 'lapse_low', 'lapse_high')
status = not_trained.aggr(status, *fields, session_date='DATE(MAX(session_start_time))')
query = (status * behavior_analysis.BehavioralSummaryByDate * subject.Death.proj('death_ts'))
df = (
    query
         .fetch(format='frame')
         .reset_index()
         .drop('subject_project', axis=1)
)
df.subject_uuid = df.subject_uuid.astype(str)
cull_reasons = load_csv('cull_reasons.csv')
df = pd.merge(df, cull_reasons, on='subject_uuid')

# NB: Untrainable training status takes precedence over cull reason
not_trained = len(mice_started_training) - len(trained)
untrainable = df['training_status'] == 'untrainable'
time_limit = (df.cull_reason == 'time limit reached') & ~untrainable
low_trial_n = df['n_trials_date'] < 400
biased = df['bias'].abs() > 15
low_perf = df['performance_easy'] < 65
# Inspecting deaths
injury = ('acute injury', 'infection or illness', 'issue during surgery', 'found dead')
# benign = ('regular experiment end', 'time limit reached', 'benign experimental impediments')


premature_death = ~untrainable & (df.cull_reason != 'time limit reached')
sick = df.training_status[df.cull_reason.isin(injury)][premature_death]
benign = premature_death & (df.cull_reason == 'benign experimental impediments')
# Unknown
ambiguous_death = premature_death & df.cull_reason.isin(('regular experiment end', None))

# Putative untrainable = marked as regular experiment end and > 40 sessions in total, not just 1a
putative_untrainable = ambiguous_death & (df['training_day'] > 40)
stats = [
    f'Out of {len(mice_started_training)} mice, {not_trained} mice had not completed training',
    f'Of those, {len(still_training_uuids)} was still in training at the time of writing',
    f'{len(df[untrainable])} mice didn’t progress to the first stage within 40 days of training',
    f'due to an extremely high bias and/or low trial count (n={len(df[untrainable & (biased | low_trial_n)])}) or ' +
    f'an otherwise low performance (n={len(df[untrainable & low_perf & ~(biased | low_trial_n)])})',
    f'An additional {sum(time_limit | putative_untrainable)} progressed further but not within the experimenter\'s time frame',
    f'{len(sick)} mice died of infection, illness or acute injury',
    f'{len(df[benign & premature_death])} could not be trained due to benign experimental impediments',
    f'For {sum(ambiguous_death & ~putative_untrainable)} mice the reason was not reported'
]
[print(s) for s in stats]
