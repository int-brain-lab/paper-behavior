"""
Decoding of lab membership based on biased psychometrics

@author: Guido Meijer
6 May 2020
"""

import seaborn as sns
import numpy as np
from os.path import join
import matplotlib.pyplot as plt
from paper_behavior_functions import (figpath, seaborn_style, group_colors,
                                      query_sessions_around_criterion, institution_map)
from ibl_pipeline import reference, subject, behavior
from dj_tools import fit_psychfunc, dj2pandas
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score, confusion_matrix

# Initialize
fig_path = figpath()
pal = group_colors()
institution_map, col_names = institution_map()
col_names = col_names[:-1]

# Settings
DECODER = 'forest'          # forest, bayes or regression
NUM_SPLITS = 3              # n in n-fold cross validation
ITERATIONS = 2000           # how often to decode
METRICS = ['threshold_l', 'threshold_r', 'bias_l', 'bias_r', 'lapselow_l', 'lapselow_r',
           'lapsehigh_l', 'lapsehigh_r']
METRICS_CONTROL = ['threshold_l', 'threshold_r', 'bias_l', 'bias_r', 'lapselow_l', 'lapselow_r',
                   'lapsehigh_l', 'lapsehigh_r', 'time_zone']
PLOT_METRICS = True
SAVE_FIG = True


# Decoding function with n-fold cross validation
def decoding(resp, labels, clf, NUM_SPLITS):
    kf = KFold(n_splits=NUM_SPLITS, shuffle=True)
    y_pred = np.array([])
    y_true = np.array([])
    for train_index, test_index in kf.split(resp):
        train_resp = resp[train_index]
        test_resp = resp[test_index]
        clf.fit(train_resp, [labels[j] for j in train_index])
        y_pred = np.append(y_pred, clf.predict(test_resp))
        y_true = np.append(y_true, [labels[j] for j in test_index])
    f1 = f1_score(y_true, y_pred, labels=np.unique(labels), average='micro')
    unique_labels, label_counts = np.unique(labels, return_counts=True)
    cm = confusion_matrix(y_true, y_pred, labels=unique_labels)
    return f1, cm


# %% query sessions
use_sessions = query_sessions_around_criterion(criterion='ephys', days_from_criterion=[2, 0])[0]
b = (use_sessions * subject.Subject * subject.SubjectLab * reference.Lab * behavior.TrialSet.Trial
     & 'task_protocol LIKE "%biased%"')

# load data into pandas dataframe
b2 = b.proj('institution_short', 'subject_nickname', 'task_protocol',
            'trial_stim_contrast_left', 'trial_stim_contrast_right', 'trial_response_choice',
            'task_protocol', 'trial_stim_prob_left', 'trial_feedback_type',
            'trial_response_time', 'trial_stim_on_time', 'time_zone')
bdat = b2.fetch(order_by='institution_short, subject_nickname, session_start_time, trial_id',
                format='frame').reset_index()
behav = dj2pandas(bdat)
behav['institution_code'] = behav.institution_short.map(institution_map)

# exclude contrasts that were part of a pilot with a different contrast set
behav = behav[((behav['signed_contrast'] != -8) & (behav['signed_contrast'] != -4)
               & (behav['signed_contrast'] != 4) & (behav['signed_contrast'] != 8))]

biased_fits = pd.DataFrame()
for i, nickname in enumerate(behav['subject_nickname'].unique()):
    if np.mod(i+1, 10) == 0:
        print('Processing data of subject %d of %d' % (i+1,
                                                       len(behav['subject_nickname'].unique())))

    # Get lab and timezone
    lab = behav.loc[behav['subject_nickname'] == nickname, 'institution_code'].unique()[0]
    time_zone = behav.loc[behav['subject_nickname'] == nickname, 'time_zone'].unique()[0]
    if (time_zone == 'Europe/Lisbon') or (time_zone == 'Europe/London'):
        time_zone_number = 0
    elif time_zone == 'America/New_York':
        time_zone_number = -5
    elif time_zone == 'America/Los_Angeles':
        time_zone_number = -7

    # Fit psychometric curve
    left_fit = fit_psychfunc(behav[(behav['subject_nickname'] == nickname)
                                   & (behav['probabilityLeft'] == 80)])
    right_fit = fit_psychfunc(behav[(behav['subject_nickname'] == nickname)
                                    & (behav['probabilityLeft'] == 20)])
    fits = pd.DataFrame(data={'threshold_l': left_fit['threshold'],
                              'threshold_r': right_fit['threshold'],
                              'bias_l': left_fit['bias'],
                              'bias_r': right_fit['bias'],
                              'lapselow_l': left_fit['lapselow'],
                              'lapselow_r': right_fit['lapselow'],
                              'lapsehigh_l': left_fit['lapsehigh'],
                              'lapsehigh_r': right_fit['lapsehigh'],
                              'nickname': nickname, 'lab': lab, 'time_zone': time_zone_number})
    biased_fits = biased_fits.append(fits, sort=False)

# %% Plot metrics

if PLOT_METRICS:
    f, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(16, 4))
    lab_colors = group_colors()

    ax1.plot([10, 20], [10, 20], linestyle='dashed', color=[0.6, 0.6, 0.6])
    for i, lab in enumerate(biased_fits['lab'].unique()):
        ax1.errorbar(biased_fits.loc[biased_fits['lab'] == lab, 'threshold_l'].mean(),
                     biased_fits.loc[biased_fits['lab'] == lab, 'threshold_r'].mean(),
                     xerr=biased_fits.loc[biased_fits['lab'] == lab, 'threshold_l'].sem(),
                     yerr=biased_fits.loc[biased_fits['lab'] == lab, 'threshold_l'].sem(),
                     fmt='s', color=lab_colors[i])
    ax1.set(xlabel='80:20 block', ylabel='20:80 block', title='Threshold')

    ax2.plot([0, 0.1], [0, 0.1], linestyle='dashed', color=[0.6, 0.6, 0.6])
    for i, lab in enumerate(biased_fits['lab'].unique()):
        ax2.errorbar(biased_fits.loc[biased_fits['lab'] == lab, 'lapselow_l'].mean(),
                     biased_fits.loc[biased_fits['lab'] == lab, 'lapselow_r'].mean(),
                     xerr=biased_fits.loc[biased_fits['lab'] == lab, 'lapselow_l'].sem(),
                     yerr=biased_fits.loc[biased_fits['lab'] == lab, 'lapselow_r'].sem(),
                     fmt='s', color=lab_colors[i])
    ax2.set(xlabel='80:20 block', ylabel='20:80 block', title='Lapse left')

    ax3.plot([0, 0.1], [0, 0.1], linestyle='dashed', color=[0.6, 0.6, 0.6])
    for i, lab in enumerate(biased_fits['lab'].unique()):
        ax3.errorbar(biased_fits.loc[biased_fits['lab'] == lab, 'lapsehigh_l'].mean(),
                     biased_fits.loc[biased_fits['lab'] == lab, 'lapsehigh_r'].mean(),
                     xerr=biased_fits.loc[biased_fits['lab'] == lab, 'lapsehigh_l'].sem(),
                     yerr=biased_fits.loc[biased_fits['lab'] == lab, 'lapsehigh_l'].sem(),
                     fmt='s', color=lab_colors[i])
    ax3.set(xlabel='80:20 block', ylabel='20:80 block', title='Lapse right')

    ax4.plot([-10, 10], [-10, 10], linestyle='dashed', color=[0.6, 0.6, 0.6])
    for i, lab in enumerate(biased_fits['lab'].unique()):
        ax4.errorbar(biased_fits.loc[biased_fits['lab'] == lab, 'bias_l'].mean(),
                     biased_fits.loc[biased_fits['lab'] == lab, 'bias_r'].mean(),
                     xerr=biased_fits.loc[biased_fits['lab'] == lab, 'bias_l'].sem(),
                     yerr=biased_fits.loc[biased_fits['lab'] == lab, 'bias_l'].sem(),
                     fmt='s', color=lab_colors[i])
    ax4.set(xlabel='80:20 block', ylabel='20:80 block', title='Bias')

    plt.tight_layout(pad=2)
    seaborn_style()
    plt.savefig(join(fig_path, 'figure4_metrics_per_lab_error_biased.pdf'), dpi=300)
    plt.savefig(join(fig_path, 'figure4_metrics_per_lab_error_biased.png'), dpi=300)

    f, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(16, 4))
    lab_colors = group_colors()

    ax1.plot([0, 30], [0, 30], linestyle='dashed', color=[0.6, 0.6, 0.6])
    sns.scatterplot(x='threshold_l', y='threshold_r', hue='lab', data=biased_fits,
                    palette=lab_colors, legend=False, ax=ax1)
    ax1.set(xlabel='80:20 block', ylabel='20:80 block', title='Threshold',
            ylim=[0, 30], xlim=[0, 30])

    ax2.plot([0, 0.2], [0, 0.2], linestyle='dashed', color=[0.6, 0.6, 0.6])
    sns.scatterplot(x='lapselow_l', y='lapselow_r', hue='lab', data=biased_fits,
                    palette=lab_colors, legend=False, ax=ax2)
    ax2.set(xlabel='80:20 block', ylabel='20:80 block', title='Lapse left',
            ylim=[0, 0.16], xlim=[0, 0.16])

    ax3.plot([0, 0.2], [0, 0.2], linestyle='dashed', color=[0.6, 0.6, 0.6])
    sns.scatterplot(x='lapsehigh_l', y='lapsehigh_r', hue='lab', data=biased_fits,
                    palette=lab_colors, legend=False, ax=ax3)
    ax3.set(xlabel='80:20 block', ylabel='20:80 block', title='Lapse right',
            ylim=[0, 0.16], xlim=[0, 0.16])

    ax4.plot([0, 0], [-20, 20], linestyle='dashed', color=[0.6, 0.6, 0.6])
    ax4.plot([-20, 20], [0, 0], linestyle='dashed', color=[0.6, 0.6, 0.6])
    sns.scatterplot(x='bias_l', y='bias_r', hue='lab', data=biased_fits,
                    palette=lab_colors, legend=False, ax=ax4)
    ax4.set(xlabel='80:20 block', ylabel='20:80 block', title='Bias',
            ylim=[-20, 20], xlim=[-20, 20])

    plt.tight_layout(pad=2)
    seaborn_style()
    plt.savefig(join(fig_path, 'figure4_metrics_per_lab_scatter_biased.pdf'), dpi=300)
    plt.savefig(join(fig_path, 'figure4_metrics_per_lab_scatter_biased.png'), dpi=300)

# %% Do decoding

# Initialize decoders
print('\nDecoding of lab membership..')
if DECODER == 'forest':
    clf = RandomForestClassifier(n_estimators=100)
elif DECODER == 'bayes':
    clf = GaussianNB()
elif DECODER == 'regression':
    clf = LogisticRegression(solver='liblinear', multi_class='auto')
else:
    raise Exception('DECODER must be forest, bayes or regression')

# Perform decoding of lab membership
result = pd.DataFrame(columns=['original', 'original_shuffled', 'confusion_matrix',
                               'control', 'control_shuffled', 'control_cm'])
decod = biased_fits.copy()
decoding_set = decod[METRICS].values
control_set = decod[METRICS_CONTROL].values
for i in range(ITERATIONS):
    if np.mod(i+1, 100) == 0:
        print('Iteration %d of %d' % (i+1, ITERATIONS))

    # Original dataset
    result.loc[i, 'original'], conf_matrix = decoding(decoding_set, list(decod['lab']),
                                                      clf, NUM_SPLITS)
    result.loc[i, 'confusion_matrix'] = conf_matrix / conf_matrix.sum(axis=1)[:, np.newaxis]
    result.loc[i, 'original_shuffled'] = decoding(decoding_set, list(decod['lab'].sample(frac=1)),
                                                  clf, NUM_SPLITS)[0]

    # Positive control dataset
    result.loc[i, 'control'], conf_matrix = decoding(control_set, list(decod['lab']),
                                                     clf, NUM_SPLITS)
    result.loc[i, 'control_cm'] = conf_matrix / conf_matrix.sum(axis=1)[:, np.newaxis]
    result.loc[i, 'control_shuffled'] = decoding(control_set, list(decod['lab'].sample(frac=1)),
                                                 clf, NUM_SPLITS)[0]

# %%
# Calculate if decoder performs above chance
chance_level = result['original_shuffled'].mean()
significance = np.percentile(result['original'], 2.5)
sig_control = np.percentile(result['control'], 0.001)
if chance_level > significance:
    print('Classification performance not significanlty above chance')
else:
    print('Above chance classification performance!')

# Plot decoding results
f, ax1 = plt.subplots(1, 1, figsize=(4, 4))
sns.violinplot(data=pd.concat([result['original'], result['control']], axis=1),
               color=[0.6, 0.6, 0.6], ax=ax1)
ax1.plot([-1, 2], [chance_level, chance_level], 'r--')
ax1.set(ylabel='Decoding performance (F1 score)', xlim=[-0.8, 1.4], ylim=[0, 0.62],
        xticklabels=['Decoding of\nlab membership', 'Positive\ncontrol\n(w. time zone)'])
# ax1.text(0, 0.6, 'n.s.', fontsize=12, ha='center')
# ax1.text(1, 0.6, '***', fontsize=15, ha='center', va='center')
plt.text(0.7, np.mean(result['original_shuffled'])-0.04, 'Chance level', color='r')
# plt.setp(ax1.xaxis.get_majorticklabels(), rotation=40)
plt.tight_layout(pad=2)
seaborn_style()

if (DECODER == 'forest') & (SAVE_FIG is True):
    plt.savefig(join(fig_path, 'figure4_decoding_%s_biased.pdf' % DECODER), dpi=300)
    plt.savefig(join(fig_path, 'figure4_decoding_%s_biased.png' % DECODER), dpi=300)
elif SAVE_FIG is True:
    plt.savefig(join(fig_path, 'suppfig4_decoding_%s_biased.pdf' % DECODER), dpi=300)
    plt.savefig(join(fig_path, 'suppfig4_decoding_%s_biased.png' % DECODER), dpi=300)
