import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import numpy as np
import os
import time

class Trainer:
    def __init__(self, model, train_data, validation_data, loss_fn, optimizer, max_epochs, save_every, execution_mode, checkpoint_path):
        """
        Parameters:
            model:                  pytorch model
            train_data:             pytorch Dataloader containing the training data
            validation_data:        pytorch Dataloader containing the validation data or None for no evaluation during training
            loss_fn:                pytorch loss function
            optimizer:              pytorch Optimizer
            max_epochs (int):       Maximum number of epochs to run
            save_every (int):       Determines N to save a checkpoint every Nth epoch
            execution_mode (str):   Either "single_gpu" or "multi_gpu" indicating whether
                                    the training is done on a single or multiple GPUs. (multi_gpu not fully implemented!)
            checkpoint_path (str):  Path to the "checkpoint.pt" file. If the file exists, continue training from that checkpoint.
        """
        self.execution_mode = execution_mode

        if execution_mode == "single_gpu":
            self.local_rank = 0
            self.global_rank = 0
            self.model = model.to(self.local_rank)
        else:
            self.local_rank = int(os.environ["LOCAL_RANK"])
            self.global_rank = int(os.environ["RANK"])
            self.model = model.to(self.local_rank)
            self.model = DDP(self.model, device_ids=[self.local_rank])
        
        self.train_data = train_data
        self.validation_data = validation_data
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.save_every = save_every
        self.max_epochs = max_epochs
        self.epochs_run = 0
        self.training_loss = torch.zeros(max_epochs)
        self.validation_loss = torch.zeros(max_epochs)

        # For mixed precision training define:
        self.scaler = torch.cuda.amp.GradScaler()

        if os.path.exists(checkpoint_path):
            self._load_checkpoint(checkpoint_path)

    def _load_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        self.model.load_state_dict(checkpoint["MODEL_STATE"])
        self.optimizer.load_state_dict(checkpoint["OPTIMIZER_STATE"])
        self.epochs_run = checkpoint["EPOCHS_RUN"]+1
        self.training_loss = checkpoint["TRAINING_LOSS"]
        self.validation_loss = checkpoint["VALIDATION_LOSS"]
        # If max epochs has been increased since last checkpoint, grow the size of loss arrays:
        if self.max_epochs > self.training_loss.shape[0]:
            self.training_loss = torch.zeros(self.max_epochs)
            self.training_loss[:checkpoint["TRAINING_LOSS"].shape[0]] = checkpoint["TRAINING_LOSS"]
            self.validation_loss = torch.zeros(self.max_epochs)
            self.validation_loss[:checkpoint["VALIDATION_LOSS"].shape[0]] = checkpoint["VALIDATION_LOSS"]
        print(f"Loaded a checkpoint created at Epoch {self.epochs_run-1}.")
        print(f"Resuming training from Epoch {self.epochs_run}")

    def _save_checkpoint(self, epoch):
        checkpoint = {}
        if self.execution_mode == "single_gpu":
            checkpoint["MODEL_STATE"] = self.model.state_dict()
        else:
            # Model is DDP wrapped in case of multi GPU execution mode:
            checkpoint["MODEL_STATE"] = self.model.module.state_dict()
        checkpoint["OPTIMIZER_STATE"] = self.optimizer.state_dict()
        checkpoint["EPOCHS_RUN"] = epoch
        checkpoint["TRAINING_LOSS"] = self.training_loss
        checkpoint["VALIDATION_LOSS"] = self.validation_loss
        torch.save(checkpoint, "checkpoint.pt")
        print(f"Epoch {epoch} | Training checkpoint saved at checkpoint.pt")

    def _save_training_metrics(self):
        training_metrics = {}
        training_metrics["TRAINING_LOSS"] = self.training_loss
        training_metrics["VALIDATION_LOSS"] = self.validation_loss
        torch.save(training_metrics, "training_metrics.pt")
        print("Training metrics saved at training_metrics.pt")

    def _save_model_parameters(self):
        if self.execution_mode == "multi_gpu":
            # Model is DDP wrapped in case of multi GPU execution mode:
            model_parameters = self.model.module.state_dict()
        else:
            model_parameters = self.model.state_dict()
        torch.save(model_parameters, "model_parameters.pt")
        print("Model parameters saved at model_parameters.pt")

    @torch.no_grad()
    def _evaluate_model(self, epoch):
        self.model.eval()
        # Initialize array for tracking validation loss over batches:
        batch_losses = np.empty(len(self.validation_data))

        # Loop over batches of validation data:
        for batch_i, (data, targets) in enumerate(self.validation_data):
            # Extract air and surface data and move to GPU:
            data_air, data_surface = data
            data_air = data_air.to(self.local_rank)
            data_surface = data_surface.to(self.local_rank)

            # Extract air and surface targets and move to GPU:
            targets_air, targets_surface = targets
            targets_air = targets_air.to(self.local_rank)
            targets_surface = targets_surface.to(self.local_rank)

            # Make prediction:
            output = self.model((data_air, data_surface))

            # Calculate validation loss:
            loss = self.loss_fn(output[0], targets_air) + self.loss_fn(output[1], targets_surface)
            batch_losses[batch_i] = loss.item()

        # Calculate mean of validation loss over batches: 
        self.validation_loss[epoch] = torch.mean(torch.from_numpy(batch_losses))
        self.model.train()

    def _run_batch(self, data, targets):
        # Extract air and surface data and move to GPU:
        data_air, data_surface = data
        data_air = data_air.to(self.local_rank)
        data_surface = data_surface.to(self.local_rank)

        # Extract air and surface targets and move to GPU:
        targets_air, targets_surface = targets
        targets_air = targets_air.to(self.local_rank)
        targets_surface = targets_surface.to(self.local_rank)
        
        self.optimizer.zero_grad()  # TODO: Gradient accumulation?
        # Forward pass:
        with torch.cuda.amp.autocast():
            output = self.model((data_air, data_surface))
            loss = self.loss_fn(output[0], targets_air) + self.loss_fn(output[1], targets_surface)
        
        del data_air
        del data_surface
        del targets_air
        del targets_surface

        # Backward pass:
        self.scaler.scale(loss).backward()
        #torch.nn.utils.clip_grad_norm(self.model.parameters(), max_norm=1)     # TODO: Gradient clipping?
        self.scaler.step(self.optimizer)
        self.scaler.update()
        return loss.item()

    def _run_epoch(self, epoch):
        start_time = time.time()
        batch_size = len(next(iter(self.train_data))[0][0])

        # Initialize array for tracking training loss over batches:
        batch_losses = np.empty(len(self.train_data))

        # Loop over batches of training data:
        for batch_i, (data, targets) in enumerate(self.train_data):
            # Run batch and store batch loss:
            batch_loss = self._run_batch(data, targets)
            batch_losses[batch_i] = batch_loss
            print(f"Epoch {epoch} | Iteration {batch_i} | Batch Loss {batch_loss}")
        
        # Calculate mean of training loss over batches: 
        epoch_loss = torch.mean(torch.from_numpy(batch_losses))

        print(f"Epoch {epoch} took {time.time() - start_time} seconds with batch size {batch_size}")
        if self.execution_mode == "multi_gpu":
            # With multiple GPUs, communicate the loss from each process to the rank 0 process:
            dist.reduce(epoch_loss, dst=0)
        if self.global_rank == 0:
            self.training_loss[epoch] = epoch_loss

    def train(self):
        print("Cuda available: ", torch.cuda.is_available())
        for epoch in range(self.epochs_run, self.max_epochs):
            self._run_epoch(epoch)
            # Evaluate the model after every epoch:
            if self.global_rank == 0 and self.validation_data != None:
                self._evaluate_model(epoch)
            # Save checkpoint after every Nth epoch:
            if self.local_rank == 0 and epoch % self.save_every == 0:
                self._save_checkpoint(epoch)
        if self.global_rank == 0:
            self._save_checkpoint(epoch)
            self._save_model_parameters()
            self._save_training_metrics()
        print("Finished training.")
