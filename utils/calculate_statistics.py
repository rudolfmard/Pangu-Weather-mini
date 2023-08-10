import sys
import os
# Get the parent directory path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Add the parent directory to sys.path
sys.path.append(parent_dir)
import data_handler

###     Run this script to calculate mean and standard deviation values for air and surface variable needed for data normalization     ###

# Define paths to air and surface training data files:
air_data_path = "../../weather_data/air_test.pt"
surface_data_path = "../../weather_data/surface_test.pt"

# Call calculate_statistics function from the data_handler file:
data_handler.calculate_statistics(air_data_path, surface_data_path)