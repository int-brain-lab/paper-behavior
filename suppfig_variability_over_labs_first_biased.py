"""
Plotting of behavioral metrics during the full task (biased blocks) per lab

Guido Meijer
6 May 2020
"""

import seaborn as sns
import numpy as np
from os.path import join
import matplotlib.pyplot as plt
from scipy import stats
import scikit_posthocs as sp
from paper_behavior_functions import (figpath, seaborn_style, group_colors, institution_map,
                                      datapath, FIGURE_WIDTH, FIGURE_HEIGHT, QUERY, fit_psychfunc, dj2pandas)
import pandas as pd
from statsmodels.stats.multitest import multipletests

# Initialize
seaborn_style()
figpath = figpath()
pal = group_colors()
institution_map, col_names = institution_map()
col_names = col_names[:-1]

# %% Process data

if QUERY is True:
    # query sessions
    from paper_behavior_functions import query_sessions_around_criterion
    from ibl_pipeline import reference, subject, behavior
    use_sessions, _ = query_sessions_around_criterion(criterion='biased',
                                                      days_from_criterion=[-1, 3])
    use_sessions = use_sessions & 'task_protocol LIKE "%biased%"'  # only get biased sessions
    b = (use_sessions * subject.Subject * subject.SubjectLab * reference.Lab
         * behavior.TrialSet.Trial)
    b2 = b.proj('institution_short', 'subject_nickname', 'task_protocol', 'session_uuid',
                'trial_stim_contrast_left', 'trial_stim_contrast_right', 'trial_response_choice',
                'task_protocol', 'trial_stim_prob_left', 'trial_feedback_type',
                'trial_response_time', 'trial_stim_on_time')
    bdat = b2.fetch(order_by='institution_short, subject_nickname, session_start_time, trial_id',
                    format='frame').reset_index()
    behav = dj2pandas(bdat)
    behav['institution_code'] = behav.institution_short.map(institution_map)
else:
    behav = pd.read_csv(join(datapath(), 'Fig4.csv'))

biased_fits = pd.DataFrame()
for i, nickname in enumerate(behav['subject_nickname'].unique()):
    if np.mod(i+1, 10) == 0:
        print('Processing data of subject %d of %d' % (i+1,
                                                       len(behav['subject_nickname'].unique())))

    # Get lab and subject uuid
    lab = behav.loc[behav['subject_nickname'] == nickname, 'institution_code'].unique()[0]
    uuid = behav.loc[behav['subject_nickname'] == nickname, 'subject_uuid'].unique()[0]

    # Fit psychometric curve
    left_fit = fit_psychfunc(behav[(behav['subject_nickname'] == nickname)
                                   & (behav['probabilityLeft'] == 80)])
    right_fit = fit_psychfunc(behav[(behav['subject_nickname'] == nickname)
                                    & (behav['probabilityLeft'] == 20)])
    neutral_fit = fit_psychfunc(behav[(behav['subject_nickname'] == nickname)
                                    & (behav['probabilityLeft'] == 50)])
    perf_easy = (behav.loc[behav['subject_nickname'] == nickname, 'correct_easy'].mean()) * 100

    fits = pd.DataFrame(data={'perf_easy': perf_easy,
                              'threshold_l': left_fit['threshold'],
                              'threshold_r': right_fit['threshold'],
                              'threshold_n': neutral_fit['threshold'],
                              'bias_l': left_fit['bias'],
                              'bias_r': right_fit['bias'],
                              'bias_n': neutral_fit['bias'],
                              'nickname': nickname, 'lab': lab, 'subject_uuid': uuid})
    biased_fits = biased_fits.append(fits, sort=False)


# %% Statistics

stats_tests = pd.DataFrame(columns=['variable', 'test_type', 'p_value'])
posthoc_tests = {}

for i, var in enumerate(['perf_easy', 'threshold_l', 'threshold_r', 'threshold_n',
                         'bias_l', 'bias_r', 'bias_n']):

    # Remove any animals with NaNs
    test_fits = biased_fits[biased_fits[var].notnull()]

    # Test for normality
    _, normal = stats.normaltest(test_fits[var])

    if normal < 0.05:
        test_type = 'kruskal'
        test = stats.kruskal(*[group[var].values
                               for name, group in test_fits.groupby('lab')])
        if test[1] < 0.05:  # Proceed to posthocs
            posthoc = sp.posthoc_dunn(test_fits, val_col=var, group_col='lab')
        else:
            posthoc = np.nan
    else:
        test_type = 'anova'
        test = stats.f_oneway(*[group[var].values
                                for name, group in test_fits.groupby('lab')])
        if test[1] < 0.05:
            posthoc = sp.posthoc_tukey(test_fits, val_col=var, group_col='lab')
        else:
            posthoc = np.nan

    posthoc_tests['posthoc_'+str(var)] = posthoc
    stats_tests.loc[i, 'variable'] = var
    stats_tests.loc[i, 'test_type'] = test_type
    stats_tests.loc[i, 'p_value'] = test[1]

# Correct for multiple tests
stats_tests['p_value'] = multipletests(stats_tests['p_value'])[1]

# %% Prepare for plotting

# Sort by lab number
biased_fits = biased_fits.sort_values('lab')

# Convert to float
biased_fits[['perf_easy', 'bias_l', 'bias_r', 'bias_n',
             'threshold_l', 'threshold_r', 'threshold_n']] = biased_fits[
                     ['perf_easy', 'bias_l', 'bias_r', 'bias_n', 'threshold_l',
                      'threshold_r', 'threshold_n']].astype(float)

# Add all mice to dataframe seperately for plotting
learned_no_all = biased_fits.copy()
#learned_no_all.loc[learned_no_all.shape[0] + 1, 'lab'] = 'All'
learned_2 = biased_fits.copy()
learned_2['lab'] = 'All'
learned_2 = biased_fits.append(learned_2)

# %%
# Plot behavioral metrics per lab
lab_colors = group_colors()
sns.set_palette(lab_colors)
seaborn_style()

vars = ['perf_easy',
        'bias_n', 'bias_l', 'bias_r',
        'threshold_n', 'threshold_l', 'threshold_r']
ylabels =['Performance (%)\non easy trials',
          'Bias (%)\n50:50 blocks',
          'Bias (%)\n20:80 blocks',
          'Bias (%)\n80:20 blocks',
          'Contrast threshold (%)\n50:50 blocks',
          'Contrast threshold (%)\n20:80 blocks',
          'Contrast threshold (%)\n80:20 blocks']
ylims = [[70, 101], [-30, 30], [-30, 30], [-30, 30],
         [0, 45], [0, 45], [0, 45]]

plt.close('all')
for v, ylab, ylim in zip(vars, ylabels, ylims):

    f, ax = plt.subplots(1, 1, figsize=(FIGURE_WIDTH/4.5, FIGURE_HEIGHT))

    sns.swarmplot(y=v, x='lab', data=learned_no_all, hue='lab',
                  palette=lab_colors, ax=ax, marker='.')
    axbox = sns.boxplot(y=v, x='lab', data=learned_2, color='white',
                        showfliers=False, ax=ax)
    ax.set(ylabel=ylab, ylim=ylim, xlabel='')
    # [tick.set_color(lab_colors[i]) for i, tick in enumerate(ax1.get_xticklabels()[:-1])]
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=60)
    axbox.artists[-1].set_edgecolor('black')
    for j in range(5 * (len(axbox.artists) - 1), 5 * len(axbox.artists)):
        axbox.lines[j].set_color('black')
    ax.get_legend().set_visible(False)

    plt.tight_layout()
    sns.despine(trim=True)
    plt.savefig(join(figpath, 'suppfig_metrics_per_lab_first_biased_%s.pdf'%v))
    plt.savefig(join(figpath, 'suppfig_metrics_per_lab_first_biased_%s.png'%v), dpi=300)
