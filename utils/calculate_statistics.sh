#!/bin/bash

#SBATCH --time=00:20:00
#SBATCH --mem=50G
#SBATCH --job-name=calculate_statistics
#SBATCH --output=calculate_statistics.out

source activate weather_model_env

python calculate_statistics.py