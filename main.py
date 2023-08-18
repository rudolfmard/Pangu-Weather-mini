from trainer import Trainer
from model import WeatherModel
import os
import data_handler as dh
import torch
from torch.distributed import init_process_group, destroy_process_group

def main(execution_mode):
    if execution_mode == "single_gpu":
        print("Training on single GPU.")
    elif execution_mode == "multi_gpu":
        print("Training on multiple GPUs.")
        init_process_group(backend="nccl")
    else:
        raise ValueError("Invalid execution mode. Valid values are 'single_gpu' or 'multi_gpu'")
    
    # This environment variable tells PyTorch CUDA allocator not to split memory blocks larger than certain size.
    # Mitigates GPU memory fragmentation and allows the training of the full original model to fit onto one GPU.
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = 'max_split_size_mb:1024'

    # Path to checkpoint.pt to continue training from that checkpoint,
    # if checkpoint.pt does not exist training starts from scratch.
    checkpoint_path="checkpoint.pt"

    """
    Training parameters:
        learning_rate (float):   Learning rate of the training, 5e-4 in original Pangu-Weather.
        max_epochs (int):        Maximum number of epochs for training, 100 in original Pangu-Weather.
        save_every (int):        Saves a checkpoint every save_every epoch.
        batch_size (int):        Batch size of the training data, 1 in original Pangu-Weather.
    """
    learning_rate = 5e-4
    max_epochs = 10
    save_every = 2
    batch_size = 1

    """
    Model parameters:
        C (int):                Dimensionality of patch embedding of the tokens. 192 in original Pangu-Weather. Make sure C is divisible by n_heads.
        depth (list[int]):      List with length of 4, defines the number of transformer blocks in each 4 EarthSpecificLayers. [2,6,6,2] in original Pangu-Weather.
        n_heads (list[int]):    List with length of 4, defines the number of heads in transformer blocks of each 4 EarthSpecificLayers. [6, 12, 12, 6] in original Pangu-Weather.
        D (int):                Dimensionality multiplier of hidden layer in transformer MLP. 4 in original Pangu-Weather.
    """
    C = 192
    depth = [2, 6, 6, 2]
    n_heads = [6, 12, 12, 6]
    D = 4

    # Create a model object:
    model = WeatherModel(C, depth, n_heads, D, batch_size, log_GPU_mem=True)

    # Create dataloader objects for training and validation data:
    train_dataset = dh.WeatherDataset(lead_time=1, air_data_path="../weather_data/air_test.pt", surface_data_path="../weather_data/surface_test.pt")
    train_dataloader = dh.prepare_dataloader(train_dataset, batch_size, execution_mode)

    # If validation_dataloader is set to None, no validation is performed between epochs.
    validation_dataset = dh.WeatherDataset(lead_time=1, air_data_path="../weather_data/air_test_validation.pt", surface_data_path="../weather_data/surface_test_validation.pt")
    validation_dataloader = dh.prepare_dataloader(validation_dataset, batch_size, execution_mode)

    # Create loss loss function and optimizer objects:
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=3e-6)
    loss_fn = torch.nn.L1Loss()

    # Create a trainer object and train the model:
    trainer = Trainer(model, train_dataloader, validation_dataloader, loss_fn, optimizer, max_epochs, save_every, execution_mode, checkpoint_path)
    trainer.train()

    # Calculate RMSE on a batch of data:
    calculate_RMSE = True
    if calculate_RMSE:
        with torch.no_grad(): 
            device = next(model.parameters()).device
            model.eval()

            # Fetch a batch from validation dataloader:
            data, targets = next(iter(validation_dataloader))

            # Move the data to the same device as the model:
            data_air, data_surface = data
            data_air = data_air.to(device)
            data_surface = data_surface.to(device)

            targets_air, targets_surface = targets
            targets_air = targets_air.to(device)
            targets_surface = targets_surface.to(device)

            # Make prediction with the model:
            output_air, output_surface = model((data_air, data_surface))

            # Calculate RMSE of the predictions on unnormalized data:
            rmse_values = dh.RMSE((dh.unnormalize_data(output_air), dh.unnormalize_data(output_surface)), 
                                (dh.unnormalize_data(targets_air), dh.unnormalize_data(targets_surface)), save=True)

    if execution_mode == "multi_gpu":
        destroy_process_group()


if __name__ == "__main__":
    import sys
    # Read execution mode given as a commandline argument (single_gpu or multi_gpu):
    execution_mode = sys.argv[1]
    main(execution_mode)