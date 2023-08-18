#!/bin/bash

#SBATCH --time=03:00:00
#SBATCH --mem=100G
#SBATCH --job-name=model_training
#SBATCH --output=training.out
#SBATCH --gres=gpu:1
#SBATCH --constraint='volta'

module load anaconda
source activate weather_model_env

python main.py single_gpu