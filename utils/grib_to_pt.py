import sys
import os
# Get the parent directory path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Add the parent directory to sys.path
sys.path.append(parent_dir)
import data_handler

data_handler.air_grib_to_tensor("air_2010_01.grib", data_folder_path="../../weather_data/")
data_handler.surface_grib_to_tensor("surface_2010_01.grib", data_folder_path="../../weather_data/")