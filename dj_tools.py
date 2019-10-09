
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from IPython import embed as shell  # for debugging

# import wrappers etc
from ibl_pipeline.utils import psychofit as psy

# ================================================================== #
# DEFINE PSYCHFUNCFIT TO WORK WITH FACETGRID IN SEABORN
# ================================================================== #


def fit_psychfunc(df):
    choicedat = df.groupby('signed_contrast').agg(
        {'choice': 'count', 'choice2': 'mean'}).reset_index()
    pars, L = psy.mle_fit_psycho(choicedat.values.transpose(), P_model='erf_psycho_2gammas',
                                 parstart=np.array(
                                     [choicedat['signed_contrast'].mean(), 20., 0.05, 0.05]),
                                 parmin=np.array(
                                     [choicedat['signed_contrast'].min(), 0., 0., 0.]),
                                 parmax=np.array([choicedat['signed_contrast'].max(), 100., 1, 1]))
    df2 = {'bias': pars[0], 'threshold': pars[1],
           'lapselow': pars[2], 'lapsehigh': pars[3]}
    df2 = pd.DataFrame(df2, index=[0])

    # # add some stuff
    # df2['easy_correct'] = df.loc[np.abs(
    #     df['signed_contrast'] > 50), 'correct'].mean(skipna=True)
    # df2['zero_contrast'] = df.loc[np.abs(
    #     df['signed_contrast'] == 0), 'choice2'].mean(skipna=True)
    # df2['median_rt'] = df['rt'].median(skipna=True)
    # df2['mean_rt'] = df['rt'].mean(skipna=True)

    # number of trials per day
    df4 = df.groupby(['session_start_time'])['correct'].count().reset_index()
    df2['ntrials_perday'] = [df4['correct'].values]
    df2['ntrials'] = df['trial'].count()

    return df2


def plot_psychometric(x, y, subj, **kwargs):

    # summary stats - average psychfunc over observers
    df = pd.DataFrame({'signed_contrast': x, 'choice': y,
                       'choice2': y, 'subject_nickname': subj})
    df2 = df.groupby(['signed_contrast', 'subject_nickname']).agg(
        {'choice2': 'count', 'choice': 'mean'}).reset_index()
    df2.rename(columns={"choice2": "ntrials",
                        "choice": "fraction"}, inplace=True)
    df2 = df2.groupby(['signed_contrast']).mean().reset_index()
    df2 = df2[['signed_contrast', 'ntrials', 'fraction']]

    # fit psychfunc
    pars, L = psy.mle_fit_psycho(df2.transpose().values,  # extract the data from the df
                                 P_model='erf_psycho_2gammas',
                                 parstart=np.array(
                                     [df2['signed_contrast'].mean(), 15., 0.05, 0.05]),
                                 parmin=np.array(
                                     [df2['signed_contrast'].min(), 0., 0., 0.]),
                                 parmax=np.array([df2['signed_contrast'].max(), 30., 0.5, 0.5]))

    # plot psychfunc
    g = sns.lineplot(np.arange(-29, 29),
                     psy.erf_psycho_2gammas(pars, np.arange(-29, 29)), **kwargs)

    # plot psychfunc: -100
    sns.lineplot(np.arange(-37, -32),
                 psy.erf_psycho_2gammas(pars, np.arange(-103, -98)), **kwargs)
    sns.lineplot(np.arange(32, 37),
                 psy.erf_psycho_2gammas(pars, np.arange(98, 103)), **kwargs)

    # now break the x-axis
    # if 100 in df.signed_contrast.values and not 50 in df.signed_contrast.values:
    df['signed_contrast'] = df['signed_contrast'].replace(-100, -35)
    df['signed_contrast'] = df['signed_contrast'].replace(100, 35)

    df3 = df.groupby(['signed_contrast', 'subject_nickname']).agg(
        {'choice2': 'count', 'choice': 'mean'}).reset_index()

    # plot datapoints with errorbars on top
    if df['subject_nickname'].nunique() > 1:
        sns.lineplot(df3['signed_contrast'], df3['choice'], err_style="bars", 
            linewidth=0, linestyle='None', mew=0.5,
                     marker='o', ci=68, **kwargs)

    # # ADD TEXT WITH THE PSYCHOMETRIC FUNCTION PARAMETERS
    # if len(df['subject_nickname'].unique()) == 1:

    # 	try:
    # 		# add text with parameters into the plot
    # 		if kwargs['label'] == '50':
    # 			ypos = 0.5
    # 			# ADD PSYCHOMETRIC FUNCTION PARAMS
    # 			plt.text(-35, ypos, r'$\mu\/ %.2f,\/ \sigma\/ %.2f,$'%(pars[0], pars[1]) + '\n' + r'$\gamma \/%.2f,\/ \lambda\/ %.2f$'%(pars[2], pars[3]),
    # 			fontweight='normal', fontsize=5, color=kwargs['color'])

    # 		elif kwargs['label'] == '20':
    # 			ypos = 0.3
    # 			# ADD PSYCHOMETRIC FUNCTION PARAMS
    # 			plt.text(-35, ypos, r'$\mu\/ %.2f,\/ \sigma\/ %.2f,$'%(pars[0], pars[1]) + '\n' + r'$\gamma \/%.2f,\/ \lambda\/ %.2f$'%(pars[2], pars[3]),
    # 			fontweight='normal', fontsize=5, color=kwargs['color'])

    # 		elif kwargs['label'] == '80':
    # 			ypos = 0.7
    # 			# ADD PSYCHOMETRIC FUNCTION PARAMS
    # 			plt.text(-35, ypos, r'$\mu\/ %.2f,\/ \sigma\/ %.2f,$'%(pars[0], pars[1]) + '\n' + r'$\gamma \/%.2f,\/ \lambda\/ %.2f$'%(pars[2], pars[3]),
    # 			fontweight='normal', fontsize=5, color=kwargs['color'])
    # 	except: # when there is no label
    # 		# pass
    # 		ypos = 0.5
    # 		# ADD PSYCHOMETRIC FUNCTION PARAMS
    # 		plt.text(-35, ypos, r'$\mu\/ %.2f,\/ \sigma\/ %.2f,$'%(pars[0], pars[1]) + '\n' + r'$\gamma \/%.2f,\/ \lambda\/ %.2f$'%(pars[2], pars[3]),
    # 		fontweight='normal', fontsize=8, color=kwargs['color'])
    # 		# plt.text(12, 0.1, '1 mouse', fontsize=10, color='k')

    # # print the number of mice
    # if df['subject_nickname'].nunique() == 1:
    # 	# plt.text(12, 0.1, '1 mouse', fontsize=10, color='k')
    #     pass
    # else:
    # 	plt.text(12, 0.1, '%d mice'%(df['subject_nickname'].nunique()), fontsize=10, color='k')

    # if brokenXaxis:
    g.set_xticks([-35, -25, -12.5, 0, 12.5, 25, 35])
    g.set_xticklabels(['-100', '-25', '-12.5', '0', '12.5', '25', '100'],
                      size='small', rotation=45)
    g.set_xlim([-40, 40])
    g.set_ylim([0, 1])
    g.set_yticks([0, 0.25, 0.5, 0.75, 1])
    g.set_yticklabels(['0', '25', '50', '75', '100'])


def plot_chronometric(x, y, subj, **kwargs):

    df = pd.DataFrame(
        {'signed_contrast': x, 'rt': y, 'subject_nickname': subj})
    df.dropna(inplace=True)  # ignore NaN RTs
    df2 = df.groupby(['signed_contrast', 'subject_nickname']
                     ).agg({'rt': 'median'}).reset_index()
    # df2 = df2.groupby(['signed_contrast']).mean().reset_index()
    df2 = df2[['signed_contrast', 'rt', 'subject_nickname']]

    # if 100 in df.signed_contrast.values and not 50 in df.signed_contrast.values:
    df2['signed_contrast'] = df2['signed_contrast'].replace(-100, -35)
    df2['signed_contrast'] = df2['signed_contrast'].replace(100, 35)

    ax = sns.lineplot(x='signed_contrast', y='rt', err_style="bars", mew=0.5,
                 ci=68, data=df2[np.abs(df2.signed_contrast) < 35], **kwargs)
    # sns.scatterplot(x='signed_contrast', y='rt', marker='o',
    #              data=df2[np.abs(df2.signed_contrast) < 35], **kwargs)

    # all the points
    if df['subject_nickname'].nunique() > 1:
        sns.lineplot(x='signed_contrast', y='rt', err_style="bars", mew=0.5, linewidth=0,
                      marker='o', ci=68, data=df2, **kwargs)

    ax.set_xticks([-35, -25, -12.5, 0, 12.5, 25, 35])
    ax.set_xticklabels(['-100', '-25', '-12.5', '0', '12.5', '25', '100'],
                       size='small', rotation=45)

    if df['signed_contrast'].min() >= 0:
        ax.set_xlim([-5, 40])
        ax.set_xticks([0, 6, 12.5, 25, 35])
        ax.set_xticklabels(['0', '6.25', '12.5', '25', '100'],
                           size='small', rotation=45)


def add_n(x, y, sj, **kwargs):

    df = pd.DataFrame({'signed_contrast': x, 'choice': y,
                       'choice2': y, 'subject_nickname': sj})

    # ADD TEXT ABOUT NUMBER OF ANIMALS AND TRIALS
    plt.text(15, 0.2, '%d mice, %d trials' % (df.subject_nickname.nunique(), df.choice.count()),
             fontweight='normal', fontsize=6, color='k')


def dj2pandas(behav):

    # make sure all contrasts are positive
    behav['trial_stim_contrast_right'] = np.abs(
        behav['trial_stim_contrast_right'])
    behav['trial_stim_contrast_left'] = np.abs(
        behav['trial_stim_contrast_left'])

    behav['signed_contrast'] = (
        behav['trial_stim_contrast_right'] - behav['trial_stim_contrast_left']) * 100
    behav['signed_contrast'] = behav.signed_contrast.astype(int)

    behav['trial'] = behav.trial_id # for psychfuncfit
    val_map = {'CCW': 1, 'No Go': 0, 'CW': -1}
    behav['choice'] = behav['trial_response_choice'].map(val_map)
    behav['correct'] = np.where(
        np.sign(behav['signed_contrast']) == behav['choice'], 1, 0)
    behav.loc[behav['signed_contrast'] == 0, 'correct'] = np.NaN

    behav['choice_right'] = behav.choice.replace(
        [-1, 0, 1], [0, np.nan, 1])  # code as 0, 100 for percentages
    behav['choice2'] = behav.choice_right  # for psychfuncfit
    behav['correct_easy'] = behav.correct
    behav.loc[np.abs(behav['signed_contrast']) < 50, 'correct_easy'] = np.NaN
    behav.rename(
        columns={'trial_stim_prob_left': 'probabilityLeft'}, inplace=True)
    behav['probabilityLeft'] = behav['probabilityLeft'] * 100
    behav['probabilityLeft'] = behav.probabilityLeft.astype(int)

    # compute rt
    if 'trial_response_time' in behav.columns:
        behav['rt'] = behav['trial_response_time'] - behav['trial_stim_on_time']
            # ignore a bunch of things for missed trials
        # don't count RT if there was no response
        behav.loc[behav.choice == 0, 'rt'] = np.nan
        # don't count RT if there was no response
        behav.loc[behav.choice == 0, 'trial_feedback_type'] = np.nan

    # CODE FOR HISTORY
    behav['previous_choice'] = behav.choice.shift(1)
    behav.loc[behav.previous_choice == 0, 'previous_choice'] = np.nan
    behav['previous_outcome'] = behav.trial_feedback_type.shift(1)
    behav.loc[behav.previous_outcome == 0, 'previous_outcome'] = np.nan
    behav['previous_contrast'] = np.abs(behav.signed_contrast.shift(1))
    behav['previous_choice_name'] = behav['previous_choice'].map(
        {-1: 'left', 1: 'right'})
    behav['previous_outcome_name'] = behav['previous_outcome'].map(
        {-1: 'post-error', 1: 'post-correct'})
    behav['repeat'] = (behav.choice == behav.previous_choice)

    # to more easily retrieve specific training days
    behav['days'] = (behav['session_start_time'] -
                     behav['session_start_time'].min()).dt.days

    return behav
