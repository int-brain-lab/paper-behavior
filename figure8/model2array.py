"""
author: alejandro pan-vazquez
"""

import cPickle
import sys
import os
sys.path.insert(0, '/Users/alex/Documents/PYTHON/trialhistory_frund/intertrial/')
sys.path.insert(0, '/Users/alex/Documents/PYTHON/trialhistory_frund/')
import intertrial
import history
import statistics
from matplotlib import pylab as pl
import glob
import numpy as np



def model2array():
    """
    Runs on python 2.7 from command line 2 argvs -  argv[1] results , argv [2] output . Easy!
    Collects the data from each individual file from Frund et al. 2014 tooolbox and generates  a np.array compatible with
    python3+
    :return: Returns and exports np.array with kernels
    """
    path_data  =  sys.argv[1]
    path_output = sys.argv[2]

    #Start uploading results
    hf  =  history.history_impulses()
    kernels = []
    mouse_names = []
    data_files = []
    os.chdir(path_data)
    for file in glob.glob("*.pcl"):
        data_files.append(file)

    for mouse in data_files:
        results_ = cPickle.load (open(path_data + mouse, 'r' ))
        M = results_['model_w_hist']
        C  = statistics.Kernel_and_Slope_Collector(hf, M.hf0, range(1, M.hf0))
        kernels.append(C(M))
        mouse_names.append(mouse[:-8])

    kernels = pl.array(kernels).T
    cPickle.dump(kernels, open(path_output + '/all_kernels.pkl', 'w'))
    cPickle.dump(mouse_names, open(path_output + '/mouse_names.pkl', 'w'))

    kr=[]
    kz=[]

    for mouse in mouse_names:
        i = mouse_names.index(mouse)
        al = kernels[14, i]
        kr.append(kernels[:7, i] * al)
        kz.append(kernels[7:14, i] * al)

    #kr = np.hstack([np.vstack(mouse_names), np.vstack(kr)])
    #kz = np.hstack([np.vstack(mouse_names), np.vstack(kz)])
    cPickle.dump(kr, open(path_output + '/kr.pkl', 'w'))
    np.save(path_output + '/kr.npy', kr)
    cPickle.dump(kz, open(path_output + '/kz.pkl', 'w'))
    np.save(path_output + '/kz.npy', kz)
    return

if __name__ == '__main__':
    model2array()