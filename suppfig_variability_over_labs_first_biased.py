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
                                      FIGURE_WIDTH, FIGURE_HEIGHT, QUERY)
from dj_tools import fit_psychfunc, dj2pandas
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
                                                      days_from_criterion=[1, 3])
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
    behav = pd.read_csv(join('data', 'Fig4.csv'))

biased_fits = pd.DataFrame()
for i, nickname in enumerate(behav['subject_nickname'].unique()):
    if np.mod(i+1, 10) == 0:
        print('Processing data of subject %d of %d' % (i+1,
                                                       len(behav['subject_nickname'].unique())))

    # Get lab
    lab = behav.loc[behav['subject_nickname'] == nickname, 'institution_code'].unique()[0]

    # Fit psychometric curve
    left_fit = fit_psychfunc(behav[(behav['subject_nickname'] == nickname)
                                   & (behav['probabilityLeft'] == 80)])
    right_fit = fit_psychfunc(behav[(behav['subject_nickname'] == nickname)
                                    & (behav['probabilityLeft'] == 20)])
    perf_easy = (behav.loc[behav['subject_nickname'] == nickname, 'correct_easy'].mean()) * 100
    
    fits = pd.DataFrame(data={'perf_easy': perf_easy,
                              'threshold_l': left_fit['threshold'],
                              'threshold_r': right_fit['threshold'],
                              'bias_l': left_fit['bias'],
                              'bias_r': right_fit['bias'],
                              'nickname': nickname, 'lab': lab})
    biased_fits = biased_fits.append(fits, sort=False)

# %% Statistics
    
stats_tests = pd.DataFrame(columns=['variable', 'test_type', 'p_value'])
posthoc_tests = {}

for i, var in enumerate(['perf_easy', 'threshold_l', 'threshold_r', 'bias_l', 'bias_r']):
    _, normal = stats.normaltest(biased_fits[var])

    if normal < 0.05:
        test_type = 'kruskal'
        test = stats.kruskal(*[group[var].values
                               for name, group in biased_fits.groupby('lab')])
        if test[1] < 0.05:  # Proceed to posthocs
            posthoc = sp.posthoc_dunn(biased_fits, val_col=var, group_col='lab')
        else:
            posthoc = np.nan
    else:
        test_type = 'anova'
        test = stats.f_oneway(*[group[var].values
                                for name, group in biased_fits.groupby('lab')])
        if test[1] < 0.05:
            posthoc = sp.posthoc_tukey(biased_fits, val_col=var, group_col='lab')
        else:
            posthoc = np.nan

    posthoc_tests['posthoc_'+str(var)] = posthoc
    stats_tests.loc[i, 'variable'] = var
    stats_tests.loc[i, 'test_type'] = test_type
    stats_tests.loc[i, 'p_value'] = test[1]

# Correct for multiple tests
stats_tests['p_value'] = multipletests(stats_tests['p_value'])[1]

# %%
# Plot behavioral metrics per lab
f, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(1, 5, figsize=(FIGURE_WIDTH*1.1, FIGURE_HEIGHT))
lab_colors = group_colors()
sns.set_palette(lab_colors)

sns.swarmplot(y='perf_easy', x='lab', data=biased_fits, hue='lab',
              palette=lab_colors, ax=ax1, marker='.')
axbox = sns.boxplot(y='perf_easy', x='lab', data=biased_fits, color='white',
                    showfliers=False, ax=ax1)
ax1.set(ylabel='Performance (%)\n on easy trials', ylim=[70, 101], xlabel='')
# [tick.set_color(lab_colors[i]) for i, tick in enumerate(ax1.get_xticklabels()[:-1])]
plt.setp(ax1.xaxis.get_majorticklabels(), rotation=60)
axbox.artists[-1].set_edgecolor('black')
for j in range(5 * (len(axbox.artists) - 1), 5 * len(axbox.artists)):
    axbox.lines[j].set_color('black')
ax1.get_legend().set_visible(False)

sns.swarmplot(y='threshold_l', x='lab', data=biased_fits, hue='lab',
              palette=lab_colors, ax=ax2, marker='.')
axbox = sns.boxplot(y='threshold_l', x='lab', data=biased_fits, color='white',
                    showfliers=False, ax=ax2)
ax2.set(ylabel='Contrast threshold\n80:20 blocks (%)', ylim=[-1, 30], xlabel='')
# [tick.set_color(lab_colors[i]) for i, tick in enumerate(ax2.get_xticklabels()[:-1])]
plt.setp(ax2.xaxis.get_majorticklabels(), rotation=60)
axbox.artists[-1].set_edgecolor('black')
for j in range(5 * (len(axbox.artists) - 1), 5 * len(axbox.artists)):
    axbox.lines[j].set_color('black')
ax2.get_legend().set_visible(False)

sns.swarmplot(y='threshold_r', x='lab', data=biased_fits, hue='lab',
              palette=lab_colors, ax=ax3, marker='.')
axbox = sns.boxplot(y='threshold_r', x='lab', data=biased_fits, color='white', showfliers=False,
                    ax=ax3)
ax3.set(ylabel='Contrast threshold\n20:80 blocks (%)', ylim=[-1, 30], xlabel='')
# [tick.set_color(lab_colors[i]) for i, tick in enumerate(ax3.get_xticklabels()[:-1])]
plt.setp(ax3.xaxis.get_majorticklabels(), rotation=60)
axbox.artists[-1].set_edgecolor('black')
for j in range(5 * (len(axbox.artists) - 1), 5 * len(axbox.artists)):
    axbox.lines[j].set_color('black')
ax3.get_legend().set_visible(False)

sns.swarmplot(y='bias_l', x='lab', data=biased_fits, hue='lab', palette=lab_colors, marker='.',
              ax=ax4)
axbox = sns.boxplot(y='bias_l', x='lab', data=biased_fits, color='white', showfliers=False, ax=ax4)
ax4.set(ylabel='Contrast threshold \n20:80 blocks (%)', ylim=[-30, 30], xlabel='')
# [tick.set_color(lab_colors[i]) for i, tick in enumerate(ax3.get_xticklabels()[:-1])]
plt.setp(ax4.xaxis.get_majorticklabels(), rotation=60)
axbox.artists[-1].set_edgecolor('black')
for j in range(5 * (len(axbox.artists) - 1), 5 * len(axbox.artists)):
    axbox.lines[j].set_color('black')
ax4.get_legend().set_visible(False)

sns.swarmplot(y='bias_r', x='lab', data=biased_fits, hue='lab', palette=lab_colors, marker='.',
              ax=ax5)
axbox = sns.boxplot(y='bias_r', x='lab', data=biased_fits, color='white', showfliers=False, ax=ax5)
ax5.set(ylabel='Contrast threshold\n20:80 blocks (%)', ylim=[-30, 30], xlabel='')
# [tick.set_color(lab_colors[i]) for i, tick in enumerate(ax3.get_xticklabels()[:-1])]
plt.setp(ax5.xaxis.get_majorticklabels(), rotation=60)
axbox.artists[-1].set_edgecolor('black')
for j in range(5 * (len(axbox.artists) - 1), 5 * len(axbox.artists)):
    axbox.lines[j].set_color('black')
ax5.get_legend().set_visible(False)

# statistical annotation
for i, var in enumerate(['perf_easy', 'threshold_l', 'threshold_r', 'bias_l', 'bias_r']):
    def num_star(pvalue):
        if pvalue < 0.05:
            stars = '* p < 0.05'
        if pvalue < 0.01:
            stars = '** p < 0.01'
        if pvalue < 0.001:
            stars = '*** p < 0.001'
        if pvalue < 0.0001:
            stars = '**** p < 0.0001'
        return stars

    pvalue = stats_tests.loc[stats_tests['variable'] == var, 'p_value']
    if pvalue.to_numpy()[0] < 0.05:
        axes = [ax1, ax2, ax3, ax4, ax5]
        axes[i].annotate(num_star(pvalue.to_numpy()[0]),
                         xy=[0.1, 0.8], xycoords='axes fraction', fontsize=5)

plt.tight_layout()
sns.despine(trim=True)
plt.savefig(join(figpath, 'suppfig_metrics_per_lab_first_biased.pdf'))
plt.savefig(join(figpath, 'suppfig_metrics_per_lab_first_biased.png'), dpi=300)
