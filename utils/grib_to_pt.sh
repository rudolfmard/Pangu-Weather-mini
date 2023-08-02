#!/bin/bash

#SBATCH --time=02:00:00
#SBATCH --mem=300G
#SBATCH --job-name=grib_to_pt
#SBATCH --output=grib_to_pt.out

source activate weather_model_env

python grib_to_pt.py