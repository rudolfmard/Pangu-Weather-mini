#!/bin/bash

#SBATCH --time=01:00:00
#SBATCH --mem=100G
#SBATCH --job-name=load_CDS
#SBATCH --output=CDS_data_output.out

source activate weather_model_env

python load_data_from_CDS.py