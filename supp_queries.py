"""
This file contains some supplementary queries and statistics for the paper.

@author: Miles Wells
"""
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
Count the number of mice for whom the 1b criterion was not applied.  The 1b criterion was 
applied retroactively in September 2019.  Here we count the number of mice that reached 1b after 
the implementation date, plus the mice that progressed further than the 1a/b and never had a 
session labeled as 1b. 
"""
# Date at which trained_1b was implemented in DJ pipeline
DATE_IMPL = '2019-09-12'
sessions = acquisition.Session * behavior_analysis.SessionTrainingStatus()

# Trained 1b
a = ((query_subjects()
      .aggr(sessions & 'training_status LIKE "%1b"',
            date_trained='min(date(session_start_time))')
      & f'date_trained >= "{DATE_IMPL}"'))

# OR reached ready 4 ephys and had no 1b session
b = (query_subjects(criterion='ephys')
     .aggr(sessions,
           skipped_1b='SUM(training_status="trained_1b") = 0',
           date_trained='min(date(session_start_time))')
     & f'date_trained < "{DATE_IMPL}"' & 'skipped_1b = 1')

# Some mice met the 1b criteria before moving to biased choice world and before the date it was
# implemented.  These mice affectively had the 1b criteria applied to them.
"""
The 1b criterion should be met both before the mouse is moved to the biased protocol and before 
moving to the ephys rig.  Here we count the mice that met the 1b criteria before it was 
implemented.
"""
c = (  # Reached 1b before moving to biased protocol
    (query_subjects()
     .aggr(sessions & 'training_status LIKE "%1b"' & 'task_protocol LIKE "%training%"',
           date_trained='min(date(session_start_time))')
     & f'date_trained < "{DATE_IMPL}"').proj()
    &  # AND met ready4ephys before being moved to an ephys rig
    (query_subjects()
     .aggr((sessions * behavior.Settings.proj('pybpod_board'))
           & 'training_status LIKE "ready"' & 'pybpod_board NOT LIKE "%ephys%"',
           date_trained='min(date(session_start_time))')
     & f'date_trained < "{DATE_IMPL}"').proj()
)


met_1b_after = len(a)
skipped_1b = len(b)
met_1b_before = len(c)
n_trained = len(query_subjects(criterion='trained'))


print('The second set, called "1b", was introduced shortly afterwards (September 2019) and '
      f'was applied to {met_1b_after} of the {n_trained} mice. {skipped_1b + met_1b_before}'
      ' mice retroactively met these criteria.')

"""
Regardless of implementation date, let's look at the number of mice that didn't not abide the 1b 
criteria
"""
started_bias_early = (
    query_subjects()
    .aggr(sessions & 'training_status LIKE "%1a"' & 'task_protocol LIKE "%bias%"',
          date_trained='min(date(session_start_time))')
    .fetch('subject_uuid')
)
started_ephys_early = (
    query_subjects()
    .aggr((sessions * behavior.Settings.proj('pybpod_board'))
          & 'training_status LIKE "training"' & 'pybpod_board LIKE "%ephys%"',
           date_trained='min(date(session_start_time))')
    .fetch('subject_uuid')
)
ignored_1b = np.unique(np.r_[started_bias_early, started_ephys_early]).size
print(f'Regardless of implementation date, {ignored_1b} of the {n_trained}'
      'mice did not abide the 1b criteria.')


##########################################################
#   Breakdown of mice that didn't finish before cutoff   #
##########################################################

trained = query_subjects(criterion='trained')
# mice that reached trained but not ready4ephys, didn't die before the cut-off, and had fewer
# than 40 sessions (no session marked as 'untrainable')
session_training_status = acquisition.Session * behavior_analysis.SessionTrainingStatus()
trained_not_ready = (trained.aggr(session_training_status,
                                  unfinished='SUM(training_status="ready4ephys" OR '
                                             'training_status="untrainable" OR '
                                             'training_status="unbiasable") = 0')
                            .aggr(subject.Death, 'unfinished',
                                  alive='death_date IS NULL OR death_date > "%s"' % CUTOFF_DATE,
                                  keep_all_rows=True))
df = ((trained_not_ready & 'alive = True AND unfinished = True')
      * behavior_analysis.SessionTrainingStatus).fetch(format='frame')
last_session = df.groupby(level=0).tail(1)
# How many of these made it to complete protocol?
passed = (last_session['good_enough_for_brainwide_map']
          | last_session['training_status'].str.startswith('ready'))
print(f'Of the {len(passed)} mice that didn\'t complete training before analysis cutoff, '
      f'{sum(passed)} ({sum(passed) / len(passed):.1%}) went on to complete the pipeline')


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
print('Number of mice that went into training: %d' % len(mice_started_training))
print('Number of mice that are still in training (exclude from 1 and 2): %d' % len(still_training))
print('Number of mice that reached trained: %d' % len(trained))
print('Number of mice that didn''t reach trained: %d' % len(not_trained))

# Fetch list of mice that were not trained, and include the status of their final session
status = behavior_analysis.SessionTrainingStatus & behavior.CompleteTrialSession
fields = ('performance', 'performance_easy', 'threshold', 'bias', 'lapse_low', 'lapse_high')
query = (not_trained
         .aggr(status,
               session_start_time='max(session_start_time)',
               session_n='COUNT(session_start_time)')
         * status
         * subject.Death.proj('death_ts')
         * behavior_analysis.PsychResults.proj(*fields)
         * behavior.TrialSet.proj('n_trials'))
df = (
    (query
        .fetch(format='frame')
        .reset_index()
        .drop('subject_project', axis=1))
)

# Print a breakdown of final training statuses
print(df.training_status.value_counts(),'\n')

# Load the cull reasons from file.  These were not available through DJ.
df.subject_uuid = df.subject_uuid.astype(str)
cull_reasons = load_csv('cull_reasons.csv')
df = pd.merge(df, cull_reasons, on='subject_uuid')

# NB: Untrainable training status takes precedence over cull reason
not_trained = len(mice_started_training) - len(trained)
untrainable = df['training_status'] == 'untrainable'
time_limit = (df.cull_reason == 'time limit reached') & ~untrainable
low_trial_n = df['n_trials'] < 400
biased = df['bias'].abs() > 15
low_perf = df['performance_easy'] < 65
# Inspecting deaths
injury = ('acute injury', 'infection or illness', 'issue during surgery')
premature_death = ~untrainable & (df.cull_reason != 'time limit reached')
sick = df.training_status[df.cull_reason.isin(injury)][premature_death]
benign = premature_death & (df.cull_reason == 'benign experimental impediments')
# Unknown
ambiguous_death = premature_death & df.cull_reason.isin(('regular experiment end', None))

# Putative untrainable = marked as regular experiment end and > 40 in total, not just 1a
putative_untrainable = ambiguous_death & (df['session_n'] > 40)
print(
    f'Out of {len(mice_started_training)} mice, {not_trained} mice had not completed '
    f'training.\nOf those, '
    f'{len(still_training)} was still in training at the time of writing.\n'
    f'{len(df[untrainable])} mice didn’t progress to the first stage within 40 days of training '
    'due to an extremely high bias and/or low trial count '
    f'(n={len(df[untrainable & (biased | low_trial_n)])}) or an otherwise low performance '
    f'(n={len(df[untrainable & low_perf & ~(biased | low_trial_n)])}).\n'
    f'An additional {sum(time_limit | putative_untrainable)} progressed further but not within '
    f'the experimenter\'s time frame.\n'
    f'{len(sick)} mice died of infection, illness or acute injury.\n'
    f'{len(df[benign & premature_death])} could not be trained due to benign experimental '
    f'impediments.\nFor {sum(ambiguous_death & ~putative_untrainable)} '
    f'mice the reason was not reported.'
)
