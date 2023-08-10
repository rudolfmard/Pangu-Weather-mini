#!/bin/bash

#SBATCH --time=05:00:00
#SBATCH --mem=100G
#SBATCH --job-name=grib_to_pt
#SBATCH --output=grib_to_pt.out

source activate weather_model_env

python grib_to_pt.py