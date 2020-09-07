#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot the results from the classification of lab by loading in the .pkl files generated by
figure3f_decoding_lab_membership_basic and figure3f_decoding_lab_membership_full

Guido Meijer
18 Jun 2020
"""

import pandas as pd
import numpy as np
import seaborn as sns
from os.path import join
import matplotlib.pyplot as plt
from paper_behavior_functions import seaborn_style, figpath, FIGURE_WIDTH, FIGURE_HEIGHT

# Settings
FIG_PATH = figpath()
colors = [[1, 1, 1], [1, 1, 1], [0.6, 0.6, 0.6]]
seaborn_style()

for DECODER in ['bayes', 'forest', 'regression']:  

    # Load in results from csv file
    decoding_result = pd.read_pickle(join('classification_results',
                                          'classification_results_basic_%s.pkl' % DECODER))

    # Calculate if decoder performs above chance
    chance_level = decoding_result['original_shuffled'].mean()
    significance = np.percentile(decoding_result['original'], 2.5)
    sig_control = np.percentile(decoding_result['control'], 0.001)
    if chance_level > significance:
        print('\n%s classifier did not perform above chance' % DECODER)
        print('Chance level: %.2f (F1 score)' % chance_level)
    else:
        print('\n%s classifier did not perform above chance' % DECODER)    
        print('Chance level: %.2f (F1 score)' % chance_level)
    print('F1 score: %.2f ± %.3f' % (decoding_result['original'].mean(),
                                     decoding_result['original'].std()))

    # %%

    # Plot main Figure 3
    if DECODER == 'bayes':
        f, ax1 = plt.subplots(1, 1, figsize=(FIGURE_WIDTH/5, FIGURE_HEIGHT))
        sns.violinplot(data=pd.concat([decoding_result['control'],
                                       decoding_result['original_shuffled'],
                                       decoding_result['original']], axis=1),
                       palette=colors, ax=ax1)
        ax1.plot([-1, 3.5], [chance_level, chance_level], '--', color='k', zorder=-10)
        ax1.set(ylabel='Decoding (F1 score)', xlim=[-0.6, 2.6], ylim=[-0.1, 0.62])
        ax1.set_xticklabels(['Positive\ncontrol', 'Shuffle', 'Decoding\nof lab'],
                            rotation=90, ha='center')
        plt.tight_layout()
        sns.despine(trim=True)
        
        plt.savefig(join(FIG_PATH, 'figure3f_decoding.pdf'))
        plt.savefig(join(FIG_PATH, 'figure3f_decoding.png'), dpi=300)
        plt.close(f)
    
    # Plot supplementary Figure 3
    f, ax1 = plt.subplots(1, 1, figsize=(FIGURE_WIDTH/5, FIGURE_HEIGHT))
    sns.violinplot(data=pd.concat([decoding_result['control'],
                                   decoding_result['original_shuffled'],
                                   decoding_result['original']], axis=1),
                   palette=colors, ax=ax1)
    ax1.plot([-1, 3.5], [chance_level, chance_level], '--', color='k', zorder=-10)
    ax1.set(ylabel='Decoding (F1 score)', xlim=[-0.8, 2.6], ylim=[-0.1, 0.62])
    ax1.set_xticklabels(['Positive\ncontrol', 'Shuffle', 'Decoding\nof lab'],
                        rotation=90, ha='center')
    plt.tight_layout()
    sns.despine(trim=True)
    
    plt.savefig(join(FIG_PATH, 'suppfig3_decoding_%s.pdf' % DECODER))
    plt.savefig(join(FIG_PATH, 'suppfig3_decoding_%s.png' % DECODER), dpi=300)
    plt.close(f)
    
    # %%
    f, ax1 = plt.subplots(1, 1, figsize=(FIGURE_WIDTH/4, FIGURE_HEIGHT))
    n_labs = decoding_result['confusion_matrix'][0].shape[0]
    # sns.heatmap(data=decoding_result['confusion_matrix'].mean(), vmin=0, vmax=0.6)
    sns.heatmap(data=decoding_result['confusion_matrix'].mean())
    ax1.plot([0, 7], [0, 7], '--w')
    ax1.set(xticklabels=np.arange(1, n_labs + 1), yticklabels=np.arange(1, n_labs + 1),
            ylim=[0, n_labs], xlim=[0, n_labs],
            title='', ylabel='Actual lab', xlabel='Predicted lab')
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=40)
    plt.setp(ax1.yaxis.get_majorticklabels(), rotation=40)
    plt.gca().invert_yaxis()
    plt.tight_layout()

    plt.savefig(join(FIG_PATH, 'suppfig3_confusion_matrix_%s.pdf' % DECODER))
    plt.savefig(join(FIG_PATH, 'suppfig3_confusion_matrix_%s.png' % DECODER), dpi=300)
    plt.close(f)

    f, ax1 = plt.subplots(1, 1, figsize=(FIGURE_WIDTH/4, FIGURE_HEIGHT))
    # sns.heatmap(data=decoding_result['control_cm'].mean(), vmin=0, vmax=1)
    sns.heatmap(data=decoding_result['control_cm'].mean())
    ax1.plot([0, 7], [0, 7], '--w')
    ax1.set(xticklabels=np.arange(1, n_labs + 1), yticklabels=np.arange(1, n_labs + 1),
            title='', ylabel='Actual lab', xlabel='Predicted lab',
            ylim=[0, n_labs], xlim=[0, n_labs])
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=40)
    plt.setp(ax1.yaxis.get_majorticklabels(), rotation=40)
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(join(FIG_PATH, 'suppfig3_control_confusion_matrix_%s.pdf' % DECODER))
    plt.savefig(join(FIG_PATH, 'suppfig3_control_confusion_matrix_%s.png' % DECODER), dpi=300)
    plt.close(f)
