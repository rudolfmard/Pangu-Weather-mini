import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
import pygrib
import numpy as np

def air_grib_to_tensor(file_name, data_folder_path="../weather_data/"):
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

def surface_grib_to_tensor(file_name, data_folder_path="../weather_data/"):
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

def normalize_data(data):
    """
    Takes a tensor of upper air variables or surface variables and normalizes it
    across longitude and latitude, separately for each variable and pressure level.
    Input:
        data: tensor of shape (Hour, Z=13, H=1440, W=721, C=5) or (Hour, H=1440, W=721, C=4)
    Output:
        Normalized data tensor
    """

    # Calculate mean and standard deviation:
    if len(data.shape) == 5:
        # Air variables:
        mean = data.mean(dim=(0,2,3), keepdim=True)
        sd = data.std(dim=(0,2,3), keepdim=True)
    else:
        # Surface variables:
        mean = data.mean(dim=(0,1,2), keepdim=True)
        sd = data.std(dim=(0,1,2), keepdim=True)

    # TODO: Save mean and SD to scale predictions back
    return (data-mean)/sd

class WeatherDataset(Dataset):
    def __init__(self, lead_time):
        air_data = normalize_data(torch.load("../weather_data/air_test.pt"))
        surface_data = normalize_data(torch.load("../weather_data/surface_test.pt"))
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