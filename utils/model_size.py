import sys
import os
# Get the parent directory path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Add the parent directory to sys.path
sys.path.append(parent_dir)
from model import WeatherModel

model = WeatherModel()
n_parameters = sum(p.numel() for p in model.parameters())
print(f"Number of parameters: {n_parameters}")

B = n_parameters * 4  # for float32 parameters
MB = B / (1024 * 1024)
GB = B / (1024 * 1024 * 1024)
print(f"Size in Bytes: {B}\nSize in MegaBytes: {MB}\nSize in GigaBytes: {GB}")