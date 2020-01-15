# paper-behavior
This repository contains code to reproduce all figures of the behavior paper by the International Brain Laboratory. 

### Installation
1. Create a new conda environment: `conda create --name djenv`
2. `pip install datajoint`
3. `pip install ibl-pipeline`
4. `conda install seaborn`
5. clone https://github.com/int-brain-lab/IBL-pipeline and put it in the same parent folder as this repo (only for figure 1)

### Obtain a DataJoint account

### How to run the code
All the scripts start with the name of the figure they produce. To run a script you need to add the local path of the repository to your Python environment so that the functions in `paper_behavior_functions` and `dj_tools` can be found. Either change the working directory of your Python IDE to the path where you cloned the repository or run the commands `import_sys` and `sys.path.insert('~/path/to/repository')`.
All figure panels will appear in the `exported_figs` folder.

### Questions?
If you have any problems running this code, please open an issue or get in touch with the code's authors (written at the top of each script).