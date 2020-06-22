# paper-behavior
This repository contains code to reproduce all figures of the behavior paper by the International Brain Laboratory. If reusing any part of this code please cite the [bioRxiv paper](https://www.biorxiv.org/content/10.1101/2020.01.17.909838v2) in which these figures appear. 

### Installation
These instructions require anaconda (https://www.anaconda.com/distribution/#download-section) for Python 3 and git (https://git-scm.com/book/en/v2/Getting-Started-Installing-Git)

In an Anaconda prompt window:
1. Follow these instructions https://github.com/int-brain-lab/iblenv to install the unified iblenv
2. additionally, install
`pip install pycircstat`

`pip install nose`

`pip install scikit_posthocs`

`pip install lifelines`


`conda develop ./IBL-pipeline/prelim_analyses/behavioral_snapshots/`

### Obtain a DataJoint account through IBL JupyterHub
[IBL Jupyterhub](https://jupyterhub.internationalbrainlab.org) provides an online environment to explore the IBL behavior data pipeline.

1. Use your GitHub account to log in and go to the resource folder. 
2. Notebook `04-Access the database locally` provides the instruction to obtain the credentials to access the database. Copy the value of `dj.config`
3. In your local python IDE, do the following:
  a. `import datajoint as dj`
  b. set your local config variable `dj.config` with the value copied from JupyterHub
  c. `dj.config.save_local()`

You'll be able to run the code after the settings above.

### How to run the code
All the scripts start with the name of the figure they produce. The figure panels will appear in the `exported_figs` subfolder.

### Questions?
If you have any problems running this code, please open an issue or get in touch with the code's authors (written at the top of each script).
