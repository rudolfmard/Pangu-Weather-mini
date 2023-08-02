#!/bin/bash

#SBATCH --time=00:20:00
#SBATCH --mem=1G
#SBATCH --job-name=model_size
#SBATCH --output=model_size.out

source activate weather_model_env

python model_size.py