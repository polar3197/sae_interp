import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader
from omegaconf import DictConfig
import hydra
import wandb
import logging
import os
import time
from .base_trainer import BaseTrainer
from src.datasets.datasets import dataset_worker_init_fn
import torch.amp

log = logging.getLogger(__name__)

class ProbeTrainer(BaseTrainer):
    def __init__(self,
                  cfg: DictConfig,
                  # Standard HP Names
                  learning_rate: float, # Updated name
                  batch_size: int,
                  num_epochs: int,
                  # Optional Standard HPs
                  num_workers: int = 0,
                  pin_memory: bool = True,
                  # Add AMP flag
                  use_amp: bool = True
                 ):
        super().__init__(cfg)
        self.cfg = cfg # Store full config

        # Store specific training params directly
        self.learning_rate = learning_rate # Updated name
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.num_workers = num_workers
        self.pin_memory = pin_memory and (self.device != torch.device('cpu')) # Adjust pin_memory
        # Store AMP setting
        self.use_amp = use_amp and (self.device != torch.device('cpu')) # AMP only works on CUDA


        # Initialize other attributes
        self.model = None
        self.optimizers = {}
        self.criterion = None
        self.train_loader = None
        self.val_loader = None
        self.train_dataset = None
        self.val_dataset = None
        self.tasks = self._determine_tasks() # Determine tasks based on cfg
        # Initialize GradScaler if using AMP
        self.scaler = GradScaler(enabled=self.use_amp)
        # Track best race validation loss
        self.best_race_val_loss = float('inf')


    def _determine_tasks(self) -> list[str]:
        """Determines the list of tasks based on cfg.data.return_labels."""
        # Access return_labels from the stored full config
        requested_labels = self.cfg.data.get("return_labels")
        if requested_labels is None or not requested_labels:
            # If empty list [] was intended to mean "no labels", handle differently
            # For probes, we expect labels.
            raise ValueError("ProbeTrainer requires target tasks specified in `cfg.data.return_labels` (e.g., ['age', 'gender'] or ['all'])")
        # Special case ['all'] will be resolved during setup after dataset is loaded
        return requested_labels


    def _setup(self):
        """Instantiate model, optimizer, criterion, datasets, dataloaders."""
        log.info("Setting up Probe Trainer components...")
        if self.use_amp:
            log.info("Automatic Mixed Precision (AMP) enabled.")

        # --- Data ---
        log.info(f"Instantiating dataset {self.cfg.data._target_} with return_labels={self.cfg.data.return_labels}")
        try:
            # Instantiate datasets using the full config node for data
            self.train_dataset = hydra.utils.instantiate(self.cfg.data, mode='training')
            self.val_dataset = hydra.utils.instantiate(self.cfg.data, mode='validation')
            log.info(f"Datasets created: Train size={len(self.train_dataset)}, Val size={len(self.val_dataset)}")
        except Exception as e:
            log.exception(f"Failed to instantiate dataset using {self.cfg.data._target_}")
            raise


        # Resolve tasks if ['all'] was requested
        if self.tasks == ['all']:
            if hasattr(self.train_dataset, 'label_keys_to_load') and self.train_dataset.label_keys_to_load:
                self.tasks = self.train_dataset.label_keys_to_load
                log.info(f"Resolved 'all' tasks to: {self.tasks}")
                if self.wandb_run:
                    wandb.config.update({"resolved_tasks": self.tasks}, allow_val_change=True)
            else:
                raise ValueError("Requested ['all'] labels, but dataset couldn't provide the available keys.")


        # Determine input dimension from the first training sample
        try:
            first_item = self.train_dataset[0]
            if isinstance(first_item, (list, tuple)):
                sample_features = first_item[0]
            else: # Should not happen if tasks are requested
                raise ValueError("Dataset did not return expected (features, labels_dict) tuple.")

            # Corrected input dimension calculation assuming pooling will happen
            if sample_features.dim() == 3:
                 input_dim = sample_features.shape[2] # Use the last dimension if pooling [B, S, E] -> E
            elif sample_features.dim() == 2:
                 input_dim = sample_features.shape[1] # Use the feature dim if already [B, E]
            else:
                raise ValueError(f"Unexpected feature dimension {sample_features.dim()} in first sample.")

            log.info(f"Determined input dimension for probes: {input_dim}")
            if self.wandb_run:
                wandb.config.update({"input_dim": input_dim}, allow_val_change=True)
        except Exception as e:
            log.exception("Could not determine input_dim from dataset sample.")
            raise


        # --- DataLoaders ---
        # Use instance attributes
        persistent_workers = self.num_workers > 0
        worker_init = dataset_worker_init_fn if self.num_workers > 0 and hasattr(self.train_dataset, 'file_handle') else None # Pass init_fn only if workers > 0 and dataset uses file handle

        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            worker_init_fn=worker_init,
            persistent_workers=persistent_workers
        )
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            worker_init_fn=worker_init,
            persistent_workers=persistent_workers
        )
        log.info(f"Train Dataloader: {len(self.train_loader)} batches, {self.num_workers} workers, pin_memory={self.pin_memory}")
        log.info(f"Validation Dataloader: {len(self.val_loader)} batches")


        # --- Model ---
        log.info(f"Instantiating model: {self.cfg.model._target_}")
        try:
            # Helper function to get class count from config or default to None
            def get_class_count(task_name):
                return self.cfg.model.get(f"{task_name}_classes", None)

            self.model = hydra.utils.instantiate(
                self.cfg.model,
                input_dim=input_dim, # Use the determined input_dim after potential pooling
                # Pass class numbers explicitly from model config
                age_classes=get_class_count('age'),
                gender_classes=get_class_count('gender'),
                race_classes=get_class_count('race')
                # Add other tasks here if needed, following the pattern
            ).to(self.device)
            log.info(f"Model '{self.model.__class__.__name__}' instantiated and moved to {self.device}.")
        except Exception as e:
             log.exception(f"Failed to instantiate model {self.cfg.model._target_}")
             raise
        # log.debug(self.model)

        # --- Optimizers & Criterion ---
        self.optimizers = {}
        # Use self.learning_rate instance attribute
        lr = self.learning_rate
        for task in self.tasks:
             classifier_attr_name = f"{task}_classifier"
             if hasattr(self.model, classifier_attr_name):
                 classifier_module = getattr(self.model, classifier_attr_name)
                 # Check if input dimension matches the determined dimension
                 if hasattr(classifier_module, 'in_features') and classifier_module.in_features != input_dim:
                     log.warning(f"Classifier '{classifier_attr_name}' in_features ({classifier_module.in_features}) != determined input_dim ({input_dim}).")
                     # Optionally raise error or attempt to resize? For now, just warn.
                 self.optimizers[task] = optim.Adam(classifier_module.parameters(), lr=lr) # Use standard lr name here
                 log.info(f"Created Adam optimizer for task '{task}' (lr={lr})")
             else:
                 log.warning(f"Model {self.model.__class__.__name__} does not have attribute '{classifier_attr_name}' for task '{task}'. Optimizer not created.")

        if not self.optimizers:
            raise ValueError(f"No optimizers created. Check if tasks {self.tasks} have corresponding attributes in model.")

        self.criterion = nn.CrossEntropyLoss()
        log.info(f"Criterion: CrossEntropyLoss")


    def _train_epoch(self, epoch: int):
        """Runs one training epoch."""
        self.model.train()
        task_losses = {task: 0.0 for task in self.tasks}
        # Initialize accumulators for training accuracy
        task_correct_train = {task: 0 for task in self.tasks}
        total_samples_train = 0
        num_batches = len(self.train_loader)

        for batch_idx, batch_data in enumerate(self.train_loader):
            if not self.tasks: break
            if not isinstance(batch_data, (list, tuple)) or len(batch_data) != 2:
                 log.error(f"Train: Unexpected batch format {type(batch_data)}.")
                 continue

            features, labels_dict = batch_data
            features = features.to(self.device)
            device_labels = {task: labels_dict[task].to(self.device) for task in self.tasks if task in labels_dict}

            if len(device_labels) != len(self.tasks):
                log.warning(f"Train: Batch missing labels. Expected {self.tasks}, got {list(labels_dict.keys())}. Skipping.")
                continue

            batch_size_current = features.size(0) # Get current batch size

            # --- Apply Mean Pooling ---
            # Assuming features shape is [B, S, E], pool over sequence dim (1) -> [B, E]
            if features.dim() == 3:
                features = torch.mean(features, dim=1)
            elif features.dim() != 2:
                log.error(f"Train: Unexpected feature dimension {features.dim()}. Expected 2 or 3. Skipping batch.")
                continue
            # --------------------------

            # Common forward pass (potentially under autocast)
            try:
                # Use updated torch.amp.autocast syntax
                with torch.amp.autocast(device_type=self.device.type, enabled=self.use_amp):
                    model_outputs = self.model(features) # Now features should be [B, E]
                    if not isinstance(model_outputs, tuple) or len(model_outputs) != len(self.tasks):
                        raise TypeError(f"Model output mismatch. Expected Tuple length {len(self.tasks)}, got {type(model_outputs)} len {len(model_outputs)}")
            except Exception as e:
                # Catch potential shape mismatches here if pooling wasn't enough or classifier in_features is wrong
                log.exception(f"Train: Error during model forward pass (Input shape: {features.shape}): {e}")
                continue

            # Process each task's loss and backward pass
            batch_correct_counts = {task: 0 for task in self.tasks} # Track batch accuracy locally
            for i, task in enumerate(self.tasks):
                if task not in self.optimizers: continue

                optimizer = self.optimizers[task]
                optimizer.zero_grad() # Zero grad specific optimizer before calculating its loss

                try:
                    # Use updated torch.amp.autocast syntax around loss calculation as well
                    with torch.amp.autocast(device_type=self.device.type, enabled=self.use_amp):
                        preds = model_outputs[i]
                        labels = device_labels[task]
                        loss = self.criterion(preds, labels)

                    # Scale the loss and compute gradients
                    self.scaler.scale(loss).backward() # Remove retain_graph

                    # Unscale gradients and step optimizer
                    self.scaler.step(optimizer)

                    # Update the scale for next iteration - crucial step
                    self.scaler.update()

                    if task in task_losses: # Ensure task exists if loop continues after error
                        task_losses[task] += loss.item()

                    # Calculate batch accuracy (after step)
                    with torch.no_grad(): # No need for gradients here
                        _, predicted_labels = torch.max(preds.detach(), 1)
                        correct_count = (predicted_labels == labels).sum().item()
                        batch_correct_counts[task] = correct_count # Store batch correct count

                except (IndexError, KeyError) as e:
                    log.error(f"Train: Error accessing preds/labels for task '{task}': {e}")
                    # Skip accuracy update for this task in this batch
                    batch_correct_counts.pop(task, None) # Remove task if error
                    continue
                except RuntimeError as e:
                    # More specific logging for shape errors during criterion/backward/step
                    log.exception(f"Train: Runtime error during backward/step for task '{task}' (Preds shape: {preds.shape}, Labels shape: {labels.shape if 'labels' in locals() else 'N/A'}): {e}")
                    # Skip accuracy update for this task in this batch
                    batch_correct_counts.pop(task, None) # Remove task if error
                    continue # Skip to next task or batch if one task fails

            # Accumulate total samples and correct counts *after* processing all tasks for the batch
            total_samples_train += batch_size_current
            for task, count in batch_correct_counts.items():
                if task in task_correct_train: # Check again in case of earlier error
                    task_correct_train[task] += count


        # --- Logging at end of epoch ---
        log_data = {"epoch": epoch + 1}
        log_str = f"Epoch {epoch+1} Train:"
        if total_samples_train > 0: # Check if any samples were processed
            for task in self.tasks:
                avg_loss = task_losses.get(task, 0) / num_batches if num_batches > 0 else float('nan')
                # Use total_samples_train for accuracy calculation
                accuracy = (task_correct_train.get(task, 0) / total_samples_train * 100)
                log_data[f"train/{task}_loss"] = avg_loss
                log_data[f"train/{task}_accuracy"] = accuracy
                # Format output nicely, handle potential NaN
                loss_str = f"{avg_loss:.4f}" if not torch.isnan(torch.tensor(avg_loss)) else "NaN"
                acc_str = f"{accuracy:.2f}%" if not torch.isnan(torch.tensor(accuracy)) else "NaN"
                log_str += f" {task}_loss={loss_str} {task}_acc={acc_str}"
        else:
            log_str += " No samples processed."

        log.info(log_str)
        if self.wandb_run:
            wandb.log(log_data, step=epoch+1)


    def _validate_epoch(self, epoch: int):
        """Runs one validation epoch."""
        self.model.eval()
        task_losses = {task: 0.0 for task in self.tasks}
        task_correct = {task: 0 for task in self.tasks}
        total_samples = 0
        num_batches = len(self.val_loader)
        # Initialize storage for confusion matrix data
        all_preds = {task: [] for task in self.tasks}
        all_labels = {task: [] for task in self.tasks}
        current_epoch_losses = {task: float('nan') for task in self.tasks} # Store epoch avg losses

        with torch.no_grad():
            for batch_data in self.val_loader:
                if not self.tasks: break
                if not isinstance(batch_data, (list, tuple)) or len(batch_data) != 2:
                    log.error(f"Val: Unexpected batch format {type(batch_data)}.")
                    continue

                features, labels_dict = batch_data
                features = features.to(self.device)
                device_labels = {task: labels_dict[task].to(self.device) for task in self.tasks if task in labels_dict}

                if len(device_labels) != len(self.tasks):
                    log.warning(f"Val: Batch missing labels. Expected {self.tasks}, got {list(labels_dict.keys())}.")
                    continue

                batch_size = features.size(0)
                total_samples += batch_size

                # --- Apply Mean Pooling ---
                # Assuming features shape is [B, S, E], pool over sequence dim (1) -> [B, E]
                if features.dim() == 3:
                    features = torch.mean(features, dim=1)
                elif features.dim() != 2:
                    log.error(f"Val: Unexpected feature dimension {features.dim()}. Expected 2 or 3. Skipping batch.")
                    continue
                # --------------------------

                try:
                    # Use updated torch.amp.autocast syntax for validation forward pass
                    with torch.amp.autocast(device_type=self.device.type, enabled=self.use_amp):
                        model_outputs = self.model(features) # Now features should be [B, E]
                        if not isinstance(model_outputs, tuple) or len(model_outputs) != len(self.tasks):
                             raise TypeError(f"Val: Model output mismatch. Expected Tuple length {len(self.tasks)}, got {type(model_outputs)} len {len(model_outputs)}")

                        # Calculate losses within the autocast context
                        batch_task_losses = {}
                        for i, task in enumerate(self.tasks):
                            if task in device_labels:
                                preds = model_outputs[i]
                                labels = device_labels[task]
                                loss = self.criterion(preds, labels)
                                batch_task_losses[task] = loss.item()
                            else:
                                log.warning(f"Val: Missing label for task '{task}' in batch, loss not calculated.")

                except Exception as e:
                    log.exception(f"Val: Error during model forward pass or loss calculation (Input shape: {features.shape}): {e}")
                    continue

                # Accumulate losses and calculate accuracy (outside autocast)
                for i, task in enumerate(self.tasks):
                    if task in batch_task_losses:
                        # Accumulate weighted loss (loss * batch_size)
                        task_losses[task] += batch_task_losses[task] * batch_size

                    # Calculate accuracy and store preds/labels for confusion matrix
                    try:
                        preds = model_outputs[i] # These might be FP16 if AMP is active
                        labels = device_labels[task]
                        _, predicted_labels = torch.max(preds.detach(), 1) # Use detach

                        if task in task_correct: # Check task exists before accumulating
                            task_correct[task] += (predicted_labels == labels).sum().item()
                        if task in all_preds and task in all_labels: # Check task exists before appending
                            all_preds[task].extend(predicted_labels.cpu().tolist())
                            all_labels[task].extend(labels.cpu().tolist())

                    except (IndexError, KeyError) as e:
                         log.error(f"Val: Error accessing preds/labels for accuracy/CM task '{task}': {e}")
                         continue

        # --- Logging and Checkpointing at end of epoch ---
        log_data = {"epoch": epoch + 1}
        log_str = f"Epoch {epoch+1} Val:"
        current_race_val_loss = float('inf') # Initialize for current epoch check

        if total_samples > 0:
            for task in self.tasks:
                 # Calculate average loss and accuracy for the epoch
                 avg_loss = task_losses.get(task, 0) / total_samples if task in task_losses else float('nan')
                 accuracy = (task_correct.get(task, 0) / total_samples * 100) if task in task_correct else float('nan')
                 log_data[f"val/{task}_loss"] = avg_loss
                 log_data[f"val/{task}_accuracy"] = accuracy
                 current_epoch_losses[task] = avg_loss # Store avg loss for checkpointing check

                 # Format output nicely, handle potential NaN
                 loss_str = f"{avg_loss:.4f}" if not torch.isnan(torch.tensor(avg_loss)) else "NaN"
                 acc_str = f"{accuracy:.2f}%" if not torch.isnan(torch.tensor(accuracy)) else "NaN"
                 log_str += f" {task}_loss={loss_str} {task}_acc={acc_str}"

            # Get current race loss specifically
            current_race_val_loss = current_epoch_losses.get('race', float('inf'))

        else:
             log_str += " No samples processed."

        log.info(log_str)

        # --- WandB Logging and Checkpointing ---
        if self.wandb_run:
             wandb.log(log_data, step=epoch+1)

             # Log Confusion Matrices
             for task in self.tasks:
                  if task in all_preds and all_preds[task] and task in all_labels and all_labels[task]:
                       try:
                            class_names = None
                            # Attempt to get class names from dataset
                            if hasattr(self.val_dataset, 'get_class_names') and callable(self.val_dataset.get_class_names):
                                class_names = self.val_dataset.get_class_names(task)

                            # Fallback: get num_classes from model config and generate default names
                            if class_names is None:
                                 num_classes_cfg_key = f"{task}_classes"
                                 num_classes = self.cfg.model.get(num_classes_cfg_key)
                                 if num_classes is not None:
                                      class_names = [str(i) for i in range(num_classes)]
                                 else:
                                      log.warning(f"Could not determine class names or number of classes for task '{task}' for confusion matrix.")

                            if class_names is not None:
                                wandb.log({f"val/{task}_confusion_matrix": wandb.plot.confusion_matrix(
                                             preds=all_preds[task],
                                             y_true=all_labels[task],
                                             class_names=class_names
                                           )}, step=epoch+1)

                       except Exception as e:
                            log.error(f"Failed to log confusion matrix for task '{task}': {e}")

        # --- Save Best Race Model Checkpoint ---
        if 'race' in self.tasks and not torch.isnan(torch.tensor(current_race_val_loss)):
             if current_race_val_loss < self.best_race_val_loss:
                  self.best_race_val_loss = current_race_val_loss
                  log.info(f"*** New best validation loss for 'race': {self.best_race_val_loss:.4f}. Saving checkpoint... ***")
                  try:
                       self._save_checkpoint(
                           filename="lp_best.pth", # Specific name for best race model
                           epoch=epoch + 1,
                           race_val_loss=self.best_race_val_loss # Save the specific loss
                           # Include other relevant info if needed
                       )
                  except Exception as e:
                       log.exception("Failed to save best race model checkpoint.")


    def run(self):
        """Main execution method for the ProbeTrainer."""
        try:
            self._setup_wandb()
            # Add use_amp=self.use_amp to wandb config logging if desired
            if self.wandb_run:
                 wandb.config.update({"use_amp": self.use_amp}, allow_val_change=True)
            self._setup()
        except Exception as e:
             log.exception("Setup failed. Aborting run.")
             if self.wandb_run:
                  wandb.finish(exit_code=1) # Ensure wandb finishes on setup failure
             return # Added return

        # Use self.num_epochs instance attribute
        log.info(f"Starting probe training for tasks: {self.tasks}, Epochs: {self.num_epochs}...")
        start_time = time.time()

        try: # Wrap training loop in try/finally for wandb finish
            for epoch in range(self.num_epochs): # Use instance attribute
                epoch_start_time = time.time()
                log.info(f"--- Starting Epoch {epoch+1}/{self.num_epochs} ---")
                self._train_epoch(epoch)
                self._validate_epoch(epoch)
                epoch_duration = time.time() - epoch_start_time
                # Include best race loss in epoch log
                log.info(f"--- Epoch {epoch+1} finished in {epoch_duration:.2f}s (Best Race Val Loss: {self.best_race_val_loss:.4f}) ---")

            total_duration = time.time() - start_time
            log.info(f"Training finished in {total_duration:.2f}s.")

            try:
                # Save final model checkpoint
                log.info("Saving final model checkpoint...")
                # Determine input dim for saving checkpoint
                input_dim_saved = -1
                # Helper to get input dim safely
                def get_input_dim(classifier_name):
                    if hasattr(self.model, classifier_name):
                        return getattr(self.model, classifier_name).in_features
                    return -1

                for task in self.tasks:
                    input_dim_saved = get_input_dim(f"{task}_classifier")
                    if input_dim_saved != -1:
                         break # Found input dim from first available classifier

                # Use self.num_epochs instance attribute
                self._save_checkpoint(
                    filename="probes_final.pth",
                    is_final=True,
                    epoch=self.num_epochs,
                    input_dim=input_dim_saved,
                    tasks=self.tasks
                )
            except Exception as e:
                 log.exception("Failed to save final checkpoint.")

        except Exception as e: # Catch exceptions during training loop
             log.exception(f"Exception occurred during training loop: {e}")
             # Optionally save a checkpoint on error?
             if self.wandb_run:
                 wandb.finish(exit_code=1) # Mark run as failed
             raise # Re-raise exception after logging and finishing wandb
        finally: # Ensure wandb finishes even if training completes normally
             if self.wandb_run and wandb.run is not None: # Check if wandb run active
                 wandb.finish() # Ensure wandb finishes cleanly
