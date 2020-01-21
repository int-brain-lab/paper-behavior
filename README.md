# paper-behavior
This repository contains code to reproduce all figures of the behavior paper by the International Brain Laboratory. 

### Installation
These instructions require anaconda (https://www.anaconda.com/distribution/#download-section) for Python 3 and git (https://git-scm.com/book/en/v2/Getting-Started-Installing-Git)

In an Anaconda prompt window:
1. Create a new conda environment: `conda create --name djenv`
2. `pip install datajoint`
3. `pip install ibl-pipeline`
4. `conda install seaborn`
5. `cd <directory-you-want-this-in>` create a folder for this repository and go to that directory
6. `git clone https://github.com/int-brain-lab/paper-behavior.git`
7. `git clone https://github.com/int-brain-lab/IBL-pipeline` in the same parent folder as this repo (only for figure 1)

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
All the scripts start with the name of the figure they produce. To run a script you need to add the local path of the repository to your Python environment so that the functions in `paper_behavior_functions` and `dj_tools` can be found. Either change the working directory of your Python IDE to the path where you cloned the repository or run the commands `import_sys` and `sys.path.insert('~/path/to/repository')`.
All figure panels will appear in the `exported_figs` folder.

### Questions?
If you have any problems running this code, please open an issue or get in touch with the code's authors (written at the top of each script).
