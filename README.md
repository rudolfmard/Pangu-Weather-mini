# Pangu-Weather-mini

This repository implements the Pangu-Weather model along with data and training tools to be used in Aalto Triton computing cluster.
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
