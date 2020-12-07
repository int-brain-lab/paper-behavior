"""
Psychometric functions of training mice, within and across labs

@author: Anne Urai
15 January 2020
"""
import seaborn as sns
import os
from os.path import join
import pandas as pd
import matplotlib.pyplot as plt
from paper_behavior_functions import (figpath, seaborn_style, group_colors, datapath,
                                      query_sessions_around_criterion, institution_map,
                                      FIGURE_HEIGHT, FIGURE_WIDTH, QUERY, EXAMPLE_MOUSE,
                                      plot_psychometric, dj2pandas, plot_chronometric)
# import wrappers etc
from ibl_pipeline import reference, subject, behavior

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
                                                             days_from_criterion=[2, 0],
                                                             as_dataframe=False,
                                                             force_cutoff=True)

    # list of dicts - see https://int-brain-lab.slack.com/archives/CB13FQFK4/p1607369435116300 for explanation
    sess = use_sessions.proj('task_protocol').fetch(format='frame').reset_index().to_dict('records')

    # Trial data to fetch
    trial_fields = ('trial_stim_contrast_left',
                    'trial_stim_contrast_right',
                    'trial_response_time',
                    'trial_stim_prob_left',
                    'trial_feedback_type',
                    'trial_stim_on_time',
                    'trial_response_choice')

    # Query trial data for sessions and subject name and lab info
    trials = (behavior.TrialSet.Trial & sess).proj(*trial_fields)

    # also get info about each subject
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
    behav = pd.read_csv(join(datapath(), 'Fig3.csv'))

# print some output
print(behav.sample(n=10))

# ================================= #
# PSYCHOMETRIC FUNCTIONS
# ================================= #

# how many mice are there for each lab?
N = behav.groupby(['institution_code'])['subject_nickname'].nunique().to_dict()
behav['n_mice'] = behav.institution_code.map(N)
behav['institution_name'] = behav.institution_code + '\n ' + behav.n_mice.apply(str) + ' mice'

# plot one curve for each animal, one panel per lab
plt.close('all')
fig = sns.FacetGrid(behav,
                    col="institution_code", col_wrap=7, col_order=col_names,
                    sharex=True, sharey=True, hue="subject_uuid",
                    height=FIGURE_HEIGHT, aspect=(FIGURE_WIDTH/7)/FIGURE_HEIGHT)
fig.map(plot_psychometric, "signed_contrast", "choice_right",
        "subject_nickname", color='gray', alpha=0.7)
fig.set_titles("{col_name}")

# overlay the example mouse
tmpdat = behav[behav['subject_nickname'].str.contains(EXAMPLE_MOUSE)]
plot_psychometric(tmpdat.signed_contrast, tmpdat.choice_right, tmpdat.subject_nickname,
                  color='black', ax=fig.axes[0], legend=False)

# add lab means on top
for axidx, ax in enumerate(fig.axes.flat):
    tmp_behav = behav.loc[behav.institution_name == behav.institution_name.unique()[axidx], :]
    plot_psychometric(tmp_behav.signed_contrast, tmp_behav.choice_right,
                      tmp_behav.institution_name, ax=ax, legend=False, color=pal[axidx], linewidth=2)
    ax.set_title(sorted(behav.institution_name.unique())[axidx],
                 color=pal[axidx])

fig.despine(trim=True)
fig.set_axis_labels("\u0394 Contrast (%)", 'Rightward choices (%)')
plt.tight_layout(w_pad=-1.7)
fig.savefig(os.path.join(figpath, "figure3a_psychfuncs.pdf"))
fig.savefig(os.path.join(figpath, "figure3a_psychfuncs.png"), dpi=300)
print('done')

# %%

# Plot all labs
fig, ax1 = plt.subplots(1, 1, figsize=(FIGURE_WIDTH/5, FIGURE_HEIGHT))
for i, inst in enumerate(behav.institution_code.unique()):
    tmp_behav = behav[behav['institution_code'].str.contains(inst)]
    plot_psychometric(tmp_behav.signed_contrast, tmp_behav.choice_right,
                      tmp_behav.subject_nickname, ax=ax1, legend=False, color=pal[i])
#ax1.set_title('All labs', color='k', fontweight='bold')
ax1.set_title('All labs: %d mice'%behav['subject_nickname'].nunique())
ax1.set(xlabel='\u0394 Contrast (%)', ylabel='Rightward choices (%)')
sns.despine(trim=True)
plt.tight_layout()
fig.savefig(os.path.join(figpath, "figure3b_psychfuncs_all_labs.pdf"))
fig.savefig(os.path.join(figpath, "figure3b_psychfuncs_all_labs.png"), dpi=300)

# ================================= #
# single summary panel
# ================================= #

# Plot all labs
fig, ax1 = plt.subplots(1, 2, figsize=(8, 4))
plot_psychometric(behav.signed_contrast, behav.choice_right,
                      behav.subject_nickname, ax=ax1[0], legend=False, color='k')
ax1[0].set_title('Psychometric function', color='k', fontweight='bold')
ax1[0].set(xlabel='\u0394 Contrast (%)', ylabel='Rightward choice (%)')

plot_chronometric(behav.signed_contrast, behav.rt,
                      behav.subject_nickname, ax=ax1[1], legend=False, color='k')
ax1[1].set_title('Chronometric function', color='k', fontweight='bold')
ax1[1].set(xlabel='\u0394 Contrast (%)', ylabel='Trial duration (s)', ylim=[0, 1.4])
sns.despine(trim=True)
plt.tight_layout()
fig.savefig(os.path.join(figpath, "summary_psych_chron.pdf"))
plt.show()
