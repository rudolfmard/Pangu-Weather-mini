# Pangu-Weather-mini

This repository implements the [Pangu-Weather model](https://www.nature.com/articles/s41586-023-06185-3) along with data and training tools to be used with [Aalto Triton computing cluster](https://scicomp.aalto.fi/triton/).
The four main files of this repository are the following:
- *model.py*
  
  Implements the Pangu-Weather model as a pytorch module called `WeatherModel`.
- *trainer.py*
  
  Implements a `Trainer` python class for training the model. The class is responsible for the training loop, checkpointing and saving the model parameters and training metrics.
- *data_handler.py*

  Implements functions for preprocessing the data, that is conversion of `.grib` files to `.pt` files as well as normalization of data tensors. In addition the file defines a `WeatherDataset` class to construct a custom pytorch Dataset object and a `prepare_dataloader` function for creating a pytorch dataloader object for training.
- *main.py*

  Finally, the `main.py` script is used to orchestrate the creation of the `WeatherModel`, `Trainer`, `WeatherDataset` and dataloader objects among other components needed for training the model. The hyperparameters of the training and the model are also defined in this script. This is also the script which is submitted to Slurm workload manager of the Triron computing cluster.

## Getting Started

### Conda environment
The scripts in this repository uses packages such as `pygrib` and `cdsapi` along with some more commonly used libraries. The `utils` folder contains `weather_model_env.yml` file, which is used to create a Conda environment with all the necessary packages installed.
When working on Triton, you first need to run command `module load miniconda` in terminal to enable Conda commands. Then, run `conda env create --file weather_model_env.yml` to create a Conda environment based on the YAML file. The shell scripts submitted to Slurm workload manager will take care of activating this Conda environment when running the code. Moreover, the shell scripts in this repository actually assume that there is a Conda environment named `weather_model_env`, but this can be changed in the shell scripts. Read more from the [Triton manual about Conda environments](https://scicomp.aalto.fi/triton/apps/python-conda/).

### Data normalization
The data has to be normalized before training the model. The normalization is performed automatically when creating a `WeatherDataset` object, but the mean and standard deviation of the training data are required to be calculated beforehand. To do this, download the data  (see the section "Downloading data") and run the script `caclulate_statistics.py` via `calculate_statistics.sh` shell script (found in `utils` folder). Read more about submitting jobs in Triton from the [Triton manual](https://scicomp.aalto.fi/triton/tut/serial/). The script will save the statistics into a tensor file `statistics.pt`. These statistics are also used for unnormalizing tensors when calculcating RMSE for the predictions made by the model.

### Training the model in Triton
After you have downloaded the data (see the next section), calculated the training data statistics and configured the hyperparameters in `main.py` file, you can start training the model by submitting the `run_single_gpu.sh` shell script to the Slurm workload manager, which will run the `main.py` script. This happens by running the command `sbatch run_single_gpu.sh` in terminal. Read more about submitting jobs in Triton from the [Triton manual](https://scicomp.aalto.fi/triton/tut/serial/).

Any prints and error messages are going to be written in a `.out` file by the Slurm workload manager, and should appear in the same directory after the job finishes.

## Downloading data
The data is downloaded from [Climate Data Store (CDS)](https://cds.climate.copernicus.eu/#!/home) by the Copernicus Climate Change Service. In order to download data from CDS, you will need to create an account and obtain the API key ([instructions](https://cds.climate.copernicus.eu/api-how-to)). The `utils` folder contains a script named `load_data_from_CDS.py` which uses the CDS API to download the data. The data comes in as `.grib` files, which needs to be parsed into suitable format (pytorch tensors) in order to be used with the model. The `load_data_from_CDS.py` calls functions from the `data_handler.py` file to perform the file conversion and saves the tensors into the same directory where the `.grib` files are located.

As described in the Pangu-Weather paper, the data consists of upper-air variables and surface variables.
The upper-air variables can be found [here](https://cds.climate.copernicus.eu/cdsapp#!/dataset/reanalysis-era5-pressure-levels?tab=form). Behind the link you will need to select the following:
- Product type: *reanalysis*
- Variable: *Geopotential*, *Specific humidity*, *Temperature*, *U-component of wind* and *V-component of wind*
- Pressure level: *50*, *100*, *150*, *200*, *250*, *300*, *400*, *500*, *600*, *700*, *850*, *925* and *1000*

The surface variables can be found [here](https://cds.climate.copernicus.eu/cdsapp#!/dataset/reanalysis-era5-single-levels?tab=form). Behind the link you will need to select the following:
- Product type: *reanalysis*
- Variable -> Popular: *2m temperature*, *10m u-component of wind*, *10m v-component of wind* and *Mean sea level pressure*

After selecting the variables, proceed by selecting each hour from the time frame of interest, as well as *whole available region* for the *geographical area*. Select `GRIB` as the file format and click on *Show API request*. From there, you can copy and paste the `c.retrieve(...)` code into the `load_data_from_CDS.py` file. Make sure you pass `your/path/of/choice/file_name.grib` as the last argument into the `c.retrieve()` function, and pass that same path into the grib file parser function as well.

***
### Note
Some parts of the code refer to *single_gpu* or *multi_gpu* execution modes. The *multi_gpu* execution mode has not yet been fully implemented, but running in *single_gpu* mode works nonetheless. I recommend you to just ignore all parts of the code referring to the *multi_gpu* execution mode.
***
