import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
import pygrib
import numpy as np
import json
import os

def air_grib_to_tensor(file_name, data_folder_path):
    # Open the file using pygrib:
    with pygrib.open(data_folder_path + file_name) as grib:
        # Create an empty dictionary to hold data:
        data = {}
        # Iterate over messages in grib file:
        for message in grib:
            # Extract key names:
            date = str(message.date)
            hour = '0'+str(message.hour) if message.hour < 10 else str(message.hour)
            time = date+hour
            level = message.level
            variable = message.name
            # Add data and missing keys to nested dictionary:
            if time not in data:
                data[time] = {}
            if level not in data[time]:
                data[time][level] = {}
            data[time][level][variable] = message.values

    # Get lists of sorted keys:
    times = sorted(data.keys())
    levels = sorted(next(iter(data.values())).keys())
    variables = sorted(next(iter(next(iter(data.values())).values())).keys())
    print(f"------------------------------\nTimes:\n{times}")
    print(f"------------------------------\nLevels:\n{levels}")
    print(f"------------------------------\nVariables:\n{variables}")
    # Create a multidimensional numpy array of the data:
    # Yields an array of shape: (hours, levels, variables, 721, 1440)
    data_array = np.array([[[data[t][l][v] for v in variables] for l in levels] for t in times], dtype=np.float32)
    del data
    
    # Convert to tensor and permute the dimensions into desired order (as in the paper):
    #   (hours, levels, variables, 721, 1440) -> (hours, levels, 1440, 721, variables)
    data_tensor = torch.from_numpy(data_array)
    data_tensor = data_tensor.permute(0, 1, 4, 3, 2)

    # Save the tensor:
    torch.save(data_tensor, data_folder_path + file_name.split(".")[0] + ".pt")

def surface_grib_to_tensor(file_name, data_folder_path):
    # Open the file using pygrib:
    with pygrib.open(data_folder_path + file_name) as grib:
        # Create an empty dictionary to hold data:
        data = {}
        # Iterate over messages in grib file:
        for message in grib:
            # Extract key names:
            date = str(message.date)
            hour = '0'+str(message.hour) if message.hour < 10 else str(message.hour)
            time = date+hour
            variable = message.name
            # Add data and missing keys to nested dictionary:
            if time not in data:
                data[time] = {}
            data[time][variable] = message.values

    # Get lists of sorted keys:
    times = sorted(data.keys())
    variables = sorted(next(iter(data.values())).keys())
    print(f"------------------------------\nTimes:\n{times}")
    print(f"------------------------------\nVariables:\n{variables}")

    # Create a multidimensional numpy array of the data:
    # Yields an array of shape: (hours, variables, 721, 1440)
    data_array = np.array([[data[t][v] for v in variables] for t in times], dtype=np.float32)
    del data

    # Convert to tensor and permute the dimensions into desired order (as in the paper):
    #   (hours, variables, 721, 1440) -> (hours, 1440, 721, variables)
    data_tensor = torch.from_numpy(data_array)
    data_tensor = data_tensor.permute(0, 3, 2, 1)

    # Save the tensor:
    torch.save(data_tensor, data_folder_path + file_name.split(".")[0] + ".pt")

def calculate_statistics(air_data_path, surface_data_path):
    # Load the tensors:
    #    air_data shape:        (Hour, Z=13, H=1440, W=721, C=5)
    #    surface data shape:    (Hour, H=1440, W=721, C=4)
    air_data = torch.load(air_data_path)
    surface_data = torch.load(surface_data_path)

    # Create a dictionary to hold the data statistics:
    statistics = {}

    # Calculate mean and standard deviation:
    statistics["AIR_MEAN"] = air_data.mean(dim=(0,2,3), keepdim=True)
    statistics["AIR_SD"] = air_data.std(dim=(0,2,3), keepdim=True)
    statistics["SURFACE_MEAN"] = surface_data.mean(dim=(0,1,2), keepdim=True)
    statistics["SURFACE_SD"] = surface_data.std(dim=(0,1,2), keepdim=True)

    # Save the statistics dictionary to a file:
    torch.save(statistics, "../statistics.pt")
    print("Training data statistics saved at statistics.pt")

def unnormalize_data(data):
    """
    Takes a tensor of upper air variables or surface variables and unnormalizes the data to its original scale.
    Input:
        data: tensor of shape (Hour, Z=13, H=1440, W=721, C=5) for air variables or (Hour, H=1440, W=721, C=4) for surface variables.
    Output:
        Unnormalized data tensor
    """
    # Fetch device of the data to move the statistic tensors onto the same device:
    device = data.get_device()

    # Define path to the statistics dictionary containing mean and SD values for air and surface variables:
    statistics_path = "statistics.pt"

    # Load pre-computed statistics dictionary:
    statistics = torch.load(statistics_path)

    # Extract mean and standard deviation from statistics dictionary:
    if len(data.shape) == 5:
        # Air variables, statistics are tensors of shape (1, 13, 1, 1, 5):
        mean = statistics["AIR_MEAN"].to(device)
        sd = statistics["AIR_SD"].to(device)
    else:
        # Surface variables, statistics are tensors of shape (1, 1, 1, 4):
        mean = statistics["SURFACE_MEAN"].to(device)
        sd = statistics["SURFACE_SD"].to(device)
    return data*sd+mean

def normalize_data(data):
    """
    Takes a tensor of upper air variables or surface variables and normalizes it
    across longitude and latitude, separately for each variable and pressure level.
    Input:
        data: tensor of shape (Hour, Z=13, H=1440, W=721, C=5) for air variables or (Hour, H=1440, W=721, C=4) for surface variables.
    Output:
        Normalized data tensor
    """
    # Define path to the statistics dictionary containing mean and SD values for air and surface variables:
    statistics_path = "statistics.pt"

    # Load pre-computed statistics dictionary:
    statistics = torch.load(statistics_path)

    # Extract mean and standard deviation from statistics dictionary:
    if len(data.shape) == 5:
        # Air variables, statistics are tensors of shape (1, 13, 1, 1, 5):
        mean = statistics["AIR_MEAN"]
        sd = statistics["AIR_SD"]
    else:
        # Surface variables, statistics are tensors of shape (1, 1, 1, 4):
        mean = statistics["SURFACE_MEAN"]
        sd = statistics["SURFACE_SD"]
    return (data-mean)/sd

class WeatherDataset(Dataset):
    def __init__(self, lead_time, air_data_path, surface_data_path):
        """
        Parameters:
            lead_time (int):            Defines the time difference between the input and target in hours.
            air_data_path (str):        Defines the path to data file containing the upper-air variables.
            surface_data_path (str):    Defines the path to data file containing the surface variables.
        """
        air_data = normalize_data(torch.load(air_data_path))
        surface_data = normalize_data(torch.load(surface_data_path))
        self.x_air = air_data[:-lead_time]
        self.x_surface = surface_data[:-lead_time]
        self.y_air = air_data[lead_time:]
        self.y_surface = surface_data[lead_time:]
        self.n_samples = self.x_air.shape[0]

        # Sanity checks:
        assert self.x_air.shape == self.y_air.shape, "air data shape does not match with its labels"
        assert self.x_surface.shape == self.y_surface.shape, "surface data shape does not match with its labels"
        assert self.x_air.shape[0] == self.x_surface.shape[0], "number of samples in air and surface data does not match"

    def __getitem__(self, index):
        return (self.x_air[index], self.x_surface[index]), (self.y_air[index], self.y_surface[index])

    def __len__(self):
        return self.n_samples

def prepare_dataloader(dataset, batch_size, execution_mode, n_workers=1):
    if execution_mode == "single_gpu":
        # Regular dataloader for single GPU setup:
        dataloader = DataLoader(dataset, batch_size, pin_memory=True, shuffle=True, num_workers=n_workers)
    else:
        # When execution mode is multi GPU, use DistributedSampler with shuffle=False to ensure non-overlapping samples for each process: 
        dataloader = DataLoader(dataset, batch_size, pin_memory=True, shuffle=False, sampler=DistributedSampler(dataset))
    return dataloader

def RMSE(prediction, target, save):
    """
    Calculates the Root Mean Squared Error between prediction and target tensors.
    The tensors are of shape (B, 13, 1440, 721, 5) for upper-air variables and
    (B, 1440, 721, 4) for surface variables. B is for batch size, and if larger than one,
    the function calculates average of the RMSE values over the batch dimension. 

    Parameters:
        prediction (tuple(tensor)):     Predicted values of the variables as a tuple of tensors (upper-air variables, surface variables) at single time point.
        target (tuple(tensor)):         Target values of the variables as a tuple of tensors (upper-air variables, surface variables) at a single time point.
        save (bool):                    If True, saves a json file of the dictionary containing RMSE values.
    Returns:
        RMSE_values (dict):             Dictionary holding RMSE values of each variable on each pressure level.
    """
    air_prediction, surface_prediction = prediction
    air_target, surface_target = target
    assert air_prediction.shape == air_target.shape, f"Air predictions ({air_prediction.shape}) and targets ({air_target.shape}) have different shapes."
    assert surface_prediction.shape == surface_target.shape, f"Surface predictions ({surface_prediction.shape}) and targets ({surface_target.shape}) have different shapes."

    # Number of latitude and longitude coordinates: 
    N_lat = air_prediction.shape[3]
    N_lon = air_prediction.shape[2]

    # Calculate latitude weights:
    weights = np.deg2rad(np.arange(-90, 90.25, 0.25))
    weights = torch.from_numpy(N_lat*np.cos(weights)/np.sum(np.cos(weights))).view(1, 1, 721).to(air_prediction.device)

    # Specify key names:
    air_keys = ["Geopotential", "Specific humidity", "Temperature", "U-wind", "V-wind"]
    pressure_level_keys = ["50", "100", "150", "200", "250", "300", "400", "500", "600", "700", "850", "925", "1000"]
    surface_keys = ["10m U-wind", "10m V-wind", "2m temperature", "MSLP"]

    # Initialize dictionary to hold RMSE values:
    RMSE_values = {}

    # Loop over air-variables:
    for i in range(air_prediction.shape[4]):
        RMSE_values[air_keys[i]] = {}
        # Loop over pressure levels:
        for j in range(air_prediction.shape[1]):
            rmse = torch.sqrt(torch.sum(torch.pow(air_prediction[:,j,:,:,i]-air_target[:,j,:,:,i], 2)*weights, dim=(1,2))/(N_lat*N_lon))
            RMSE_values[air_keys[i]][pressure_level_keys[j]] = torch.mean(rmse).item()
    
    # Loop over surface variables:
    for k in range(surface_prediction.shape[3]):
        rmse = torch.sqrt(torch.sum(torch.pow(surface_prediction[:,:,:,k]-surface_target[:,:,:,k], 2)*weights, dim=(1,2))/(N_lat*N_lon))
        RMSE_values[surface_keys[k]] = torch.mean(rmse).item()

    if save:
        with open('RMSE.json', 'w') as file:
            json.dump(RMSE_values, file, indent=4)

    return RMSE_values