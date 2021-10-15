# paper-behavior
This repository contains code to reproduce all figures of the behavior paper by the International Brain Laboratory. If reusing any part of this code please cite the [bioRxiv paper](https://www.biorxiv.org/content/10.1101/2020.01.17.909838v2) in which these figures appear. 

### Installation
These instructions require anaconda (https://www.anaconda.com/distribution/#download-section) for Python 3 and git (https://git-scm.com/book/en/v2/Getting-Started-Installing-Git)

In an Anaconda prompt window:
1. Clone or download the paper-bahavior repository
2. Install the other dependencies by running `pip install -r requirements.txt`

To call the functions in this repo, either run python from within the `paper-bahavior` folder or
add the folder to your python path: 
```python
import sys
sys.path.extend([r'path/to/paper-behavior'])
```

### Accessing the data via DataJoint on your local machine

Before you can start using DataJoint with IBL data on your local machine, you will need to set your DataJoint credentials. You must specify a database connection to tell DataJoint where to look for IBL data, as well as grant access to these data by providing a username and password. 

Start by opening a new python script or terminal, then import DataJoint and set a few configuration options. With your python environment activated, run:

```python
import datajoint as dj
```

The database's hostname, username, and password are saved in the global variable `dj.config`. See it's contents by running the following line:

```python
dj.config
```

By default, it should look something like this:

```
{   'connection.charset': '',
    'connection.init_function': None,
    'database.host': 'localhost',
    'database.password': None,
    'database.port': 3306,
    'database.reconnect': True,
    'database.use_tls': None,
    'database.user': None,
    'display.limit': 12,
    'display.show_tuple_count': True,
    'display.width': 14,
    'enable_python_native_blobs': True,
    'fetch_format': 'array',
    'loglevel': 'INFO',
    'safemode': True}
```

You need to replace a few entries with the following values used for the public data: 

```{important}
Public IBL Credentials:

  hostname: datajoint-public.internationalbrainlab.org
  username: ibl-public
  password: ibl-public
```

The database connection is specified by the key `database.host`. Change the config using the values above for the fields `database.host`, `database.user` and `database.password`:

```python
dj.config["database.host"] = "datajoint-public.internationalbrainlab.org"
dj.config["database.user"] = "ibl-public"
dj.config["database.password"] = "ibl-public"
```

Then save the changes to a local JSON configuration file (`dj_local_conf.json`) by running:

```python
dj.config.save_local()
```

After the above step, every time you start your python kernel from a directory that contains this file, DataJoint will look for this file and load the config without having to set credentials again. If you want to set your credentials globally without having to be in the directory containing the file `dj_local_config.json`, you can do so by running the following:

```python
dj.config.save_global()
```

To test whether your credentials work, try connecting to the database by running:

```python
dj.conn()
```

You should find that DataJoint automatically connects to the database! To see which schemas you have access to, run:

```python
dj.list_schemas()
```

### How to run the code
All the scripts start with the name of the figure they produce. The figure panels will appear in the `exported_figs` subfolder.

### Load figures without DataJoint
To load the figures from data saved in local CSV files, edit line 21 of
 `paper_behavior_functions.py` so that `QUERY = False`.  When running the scripts for the first time the required data will be downloaded to ./data

### Download data without any code
To download data locally without running any code, simply load in an internet browser the link provided as the [URL](https://github.com/int-brain-lab/paper-behavior/blob/master/paper_behavior_functions.py#L27) in `paper_behavior_functions.py`.

### Questions?
If you have any problems running this code, please open an issue in the [iblenv repository](https://github.com/int-brain-lab/iblenv/issues) where we support users in using the IBL software tools.

You can read more about the [IBL dataset types](https://docs.google.com/spreadsheets/d/1ieLXRPLLSgUKcLvFkrqizfZl5HjdfE6bQ2KLBCRmjQo/edit#gid=1097679410) and [additional computations on the behavioral data](data.internationalbrainlab.org), such as training status and psychometric functions.

### Known issues
The data used in this paper have a number of issues.  The authors are confident that these issues
do not affect the results of the paper but nevertheless users should be aware of these
shortcomings.
  
1. NaN values may be found throughout the data.  These resulted from failures to either produce
 an event (for example the go cue tone wasn't played) or to record an event (e.g. the stimulus was
 produced but the photodiode failed to detect it).
2. Some events violated the task structure outlined in the paper.  For example during some sessions
 the go cue tone happened much later than the stimulus onset.  Although this conceivably
 affected the reaction time on some trials, it did not occur frequently enough to
 significantly affect the median reaction times.
