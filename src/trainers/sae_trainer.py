import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from omegaconf import DictConfig
import hydra # For instantiation
import wandb
import logging
import os
import time
from .base_trainer import BaseTrainer
from tqdm.auto import tqdm # Import tqdm
import torch.amp # Import the top-level amp module

# Import AMP components if available
try:
    from torch.cuda.amp import GradScaler, autocast
    AMP_AVAILABLE = True
except ImportError:
    AMP_AVAILABLE = False
    GradScaler = None # Define for type hints if not available

# Import dataset_worker_init_fn if needed by your datasets
from src.datasets.datasets import dataset_worker_init_fn

log = logging.getLogger(__name__)

class SAETrainer(BaseTrainer): # Inherit from BaseTrainer
    def __init__(
        self,
        cfg: DictConfig,
        # Standard HP Names
        learning_rate: float,
        batch_size: int,
        num_epochs: int,
        # SAE Specific HP
        l1: float,
        # Optional Standard HPs
        num_workers: int = 0,
        pin_memory: bool = True,
        # Optimization Flags
        amp_enabled: bool = True, # Enable AMP by default if available
        compile_enabled: bool = False, # Disabled by default
        validate_every_n_epochs: int = 1, # Validate every epoch by default
        # DDP parameters (will be set by main script)
        rank: int = 0,
        local_rank: int = 0,
        world_size: int = 1
    ):
        # Pass DDP parameters to BaseTrainer
        super().__init__(cfg, rank, local_rank, world_size)
        # self.cfg is already set by BaseTrainer
        # self.rank, self.local_rank, self.world_size, self.is_distributed, self.device are set by BaseTrainer

        # Store specific training params directly
        self.learning_rate = learning_rate
        # Adjust batch size for DDP: total batch size = batch_size * world_size
        # The config batch_size should represent per-GPU batch size
        self.per_gpu_batch_size = batch_size
        self.global_batch_size = batch_size * self.world_size
        self.num_epochs = num_epochs
        self.l1 = l1
        self.num_workers = num_workers
        # Adjust pin_memory based on device (only effective for CUDA)
        self.pin_memory = pin_memory and (self.device.type == 'cuda')

        # Store optimization flags
        self.amp_enabled = amp_enabled and AMP_AVAILABLE and (self.device.type == 'cuda')
        # Note: torch.compile compatibility with DDP and AMP can vary by version
        # self.compile_enabled = compile_enabled and hasattr(torch, 'compile')
        self.compile_enabled = False # nah bro we not compiling. it's not worth it.
        self.validate_every_n_epochs = validate_every_n_epochs

        # Initialize other attributes (model, optimizer, etc.) to None
        self.model = None
        self.optimizer = None
        self.criterion = None
        self.train_loader = None
        self.val_loader = None
        self.train_sampler = None
        self.val_sampler = None
        self.train_dataset = None
        self.val_dataset = None
        self.scaler = None # For AMP
        self.best_val_loss = float('inf') # For saving best model
        self.expected_embedding_dim = 768 # Expected dim after averaging

        if self.rank == 0:
             log.info(f"SAE Trainer initialized. Distributed: {self.is_distributed}, World Size: {self.world_size}")
             log.info(f"Per-GPU Batch Size: {self.per_gpu_batch_size}, Global Batch Size: {self.global_batch_size}")
             if self.amp_enabled:
                 log.info("AMP is enabled.")
             if self.compile_enabled:
                 log.info("torch.compile is enabled.")
             else:
                 if compile_enabled and not hasattr(torch, 'compile'):
                      log.warning("torch.compile requested but not available in this PyTorch version.")

    def _setup(self):
        """Instantiate datasets, samplers, dataloaders, model, optimizer, criterion, and apply compile/DDP/AMP."""
        if self.rank == 0:
             log.info("Setting up SAE Trainer components...")

        # --- Data ---
        if self.rank == 0: log.info(f"Instantiating dataset using config: {self.cfg.dataset._target_}")
        try:
            # Instantiate datasets on all ranks
            self.train_dataset = hydra.utils.instantiate(self.cfg.dataset, mode='training')
            self.val_dataset = hydra.utils.instantiate(self.cfg.dataset, mode='validation')
            if self.rank == 0: log.info(f"Datasets created: Train size={len(self.train_dataset)}, Val size={len(self.val_dataset)}")
        except Exception as e:
             log.exception(f"Rank {self.rank}: Failed to instantiate dataset using {self.cfg.dataset._target_}. Ensure it accepts a 'mode' argument.")
             if self.is_distributed: dist.barrier() # Wait for others before exiting
             raise

        # --- Samplers (for DDP) ---
        if self.is_distributed:
             self.train_sampler = DistributedSampler(
                 self.train_dataset, num_replicas=self.world_size, rank=self.rank, shuffle=True
             )
             self.val_sampler = DistributedSampler(
                 self.val_dataset, num_replicas=self.world_size, rank=self.rank, shuffle=False
             )
             if self.rank == 0: log.info("DistributedSamplers created.")
        else:
             self.train_sampler = None
             self.val_sampler = None

        # --- DataLoaders ---
        persistent_workers = self.num_workers > 0 and self.pin_memory # persistent_workers requires CUDA
        worker_init = dataset_worker_init_fn if self.num_workers > 0 and hasattr(self.train_dataset, 'file_handle') else None

        # Determine shuffle parameter based on DDP
        train_shuffle = not self.is_distributed # Shuffle only if not using DistributedSampler

        self.train_loader = DataLoader(
            self.train_dataset, batch_size=self.per_gpu_batch_size, # Use per-GPU batch size
            shuffle=train_shuffle, sampler=self.train_sampler, # Use sampler if distributed
            num_workers=self.num_workers, pin_memory=self.pin_memory,
            persistent_workers=persistent_workers, worker_init_fn=worker_init,
            drop_last=self.is_distributed # Good practice for DDP to ensure equal batches
        )
        self.val_loader = DataLoader(
            self.val_dataset, batch_size=self.per_gpu_batch_size, # Use per-GPU batch size
            shuffle=False, sampler=self.val_sampler, # Use sampler if distributed
            num_workers=self.num_workers, pin_memory=self.pin_memory,
            persistent_workers=persistent_workers, worker_init_fn=worker_init,
            drop_last=False # Usually not needed for validation
        )
        if self.rank == 0:
            log.info(f"Train Dataloader: {len(self.train_loader)} batches/GPU, {self.num_workers} workers, pin_memory={self.pin_memory}")
            log.info(f"Validation Dataloader: {len(self.val_loader)} batches/GPU")

        # --- Model ---
        if self.rank == 0: log.info(f"Instantiating model: {self.cfg.model._target_}")
        # Determine input_size (should be embedding dim = 768) and verify model config
        input_size_arg = {}
        model_input_size_config = getattr(self.cfg.model, 'input_size', None)

        if model_input_size_config is None:
             if self.rank == 0: log.info(f"model.input_size not set. Setting to default: {self.expected_embedding_dim}")
             input_size_arg['input_size'] = self.expected_embedding_dim
        elif model_input_size_config != self.expected_embedding_dim:
             if self.rank == 0: log.warning(f"Configured model.input_size ({model_input_size_config}) != expected ({self.expected_embedding_dim}). Using configured value.")
        else:
             if self.rank == 0: log.info(f"Using input_size from config: {model_input_size_config}")

        # Instantiate model and move to device
        try:
             # Instantiate on CPU first if using DDP to avoid CUDA OOM on rank 0? Usually okay to instantiate directly on device.
             self.model = hydra.utils.instantiate(self.cfg.model, **input_size_arg).to(self.device)
             if self.rank == 0: log.info(f"Model '{self.model.__class__.__name__}' instantiated and moved to {self.device}.")
        except Exception as e:
             log.exception(f"Rank {self.rank}: Failed to instantiate model.")
             if self.is_distributed: dist.barrier()
             raise

        # --- torch.compile (before DDP) ---
        if self.compile_enabled:
            if self.rank == 0: log.info("Compiling model with torch.compile...")
            try:
                # Compile model - consider compile mode options for speed/memory
                self.model = torch.compile(self.model)
                if self.rank == 0: log.info("Model compiled successfully.")
            except Exception as e:
                if self.rank == 0: log.warning(f"torch.compile failed: {e}. Proceeding without compilation.")
                self.compile_enabled = False # Disable if compilation fails


        # --- DDP Wrapping (after compile, before optimizer) ---
        if self.is_distributed:
            # Ensure model parameters are synchronized across ranks before DDP wraps them
            # This is usually handled internally by DDP constructor, but explicit sync is safe
            for param in self.model.parameters():
                 dist.broadcast(param.data, src=0)

            self.model = DDP(self.model, device_ids=[self.local_rank], output_device=self.local_rank)
            if self.rank == 0: log.info(f"Model wrapped with DDP.")


        # --- Optimizer & Criterion (after DDP wrapping) ---
        # Pass DDP-wrapped model parameters to optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()
        if self.rank == 0:
             log.info(f"Optimizer: Adam (lr={self.learning_rate})")
             log.info(f"Criterion: MSELoss")

        # --- AMP GradScaler ---
        if self.amp_enabled:
             # Fix: Remove the invalid device_type argument
             self.scaler = torch.amp.GradScaler()
             if self.rank == 0: log.info("AMP GradScaler initialized.")

        # Barrier to ensure all processes finished setup
        if self.is_distributed:
            log.debug(f"Rank {self.rank} waiting at setup barrier.")
            dist.barrier()
            log.debug(f"Rank {self.rank} passed setup barrier.")


    def _process_batch(self, batch_data):
        """Extracts features, potentially averages, moves to device."""
        if isinstance(batch_data, (list, tuple)):
             batch = batch_data[0]
        else:
             batch = batch_data

        # Move to device *before* processing if possible
        batch = batch.to(self.device, non_blocking=self.pin_memory)

        if batch.ndim != 3:
            # Still log warning, but maybe only on rank 0 to avoid spam?
            # if self.rank == 0:
            #      log.warning(f"Expected 3D input (B, S, E), but got shape {batch.shape}. Assuming it's correct for SAE.")
            return batch # Return directly
        else:
            # log.debug(f"Rank {self.rank}: Input batch shape: {batch.shape}. Taking mean over dim=1.")
            # batch_averaged = torch.mean(batch, dim=1)

            batch_flattened = (
                batch[:,1:,:] # don't want the first one
                .reshape(-1, 768) # hardcoded cause im lit
            ) # Shape (Batch*566 x 768)

            # log.debug(f"Rank {self.rank}: Averaged batch shape: {batch_averaged.shape}")
            # return batch_averaged
            return batch_flattened
            # Note: Averaging happens *after* moving to GPU here. Could be done before.


    def _train_epoch(self, epoch: int):
        """Runs one training epoch with DDP, AMP support and tqdm progress bar."""
        self.model.train() # Set model to training mode (matters for Dropout, BatchNorm)

        # Set epoch for distributed sampler
        if self.is_distributed and self.train_sampler:
             self.train_sampler.set_epoch(epoch)

        total_train_loss = 0.0
        total_recon_loss = 0.0
        total_l1_loss = 0.0
        processed_batches = 0 # Count batches processed on this rank

        # Wrap the data loader with tqdm (only show on rank 0)
        train_pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.num_epochs} [Train]",
                          leave=False, disable=self.rank != 0)

        for batch_idx, batch_data in enumerate(train_pbar):
            batch_processed = self._process_batch(batch_data)
            target_batch = batch_processed # Target is the same processed input for standard SAE

            # DDP typically requires grads to be cleared *before* forward pass if using gradient accumulation,
            # but here we clear before the autocast context which is standard.
            self.optimizer.zero_grad(set_to_none=True)

            # AMP context
            # Fix 3: Use torch.amp.autocast with device_type as the first arg
            with torch.amp.autocast(device_type=self.device.type, enabled=self.amp_enabled):
                # DDP forward pass
                recon, encoded = self.model(batch_processed)
                recon_loss = self.criterion(recon, target_batch)
                l1_loss = self.l1 * torch.norm(encoded, p=1, dim=-1).mean()
                loss = recon_loss + l1_loss

            # Scaled backward pass for AMP
            if self.scaler:
                self.scaler.scale(loss).backward() # DDP handles gradient sync here
                # Optional: Gradient Clipping (before step)
                # self.scaler.unscale_(self.optimizer) # Unscale grads if needed before clipping
                # torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward() # DDP handles gradient sync here
                # Optional: Gradient Clipping
                # torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()

            # Update running losses (for this rank)
            current_loss = loss.item()
            total_train_loss += current_loss
            total_recon_loss += recon_loss.item()
            total_l1_loss += l1_loss.item()
            processed_batches += 1

            # Update tqdm progress bar postfix (only shown on rank 0)
            if self.rank == 0:
                train_pbar.set_postfix(loss=f"{current_loss:.4f}")

        # --- Aggregate and Log Training Metrics ---
        if processed_batches > 0:
            avg_train_loss_rank = total_train_loss / processed_batches
            avg_recon_loss_rank = total_recon_loss / processed_batches
            avg_l1_loss_rank = total_l1_loss / processed_batches

            if self.is_distributed:
                 # Aggregate losses across ranks for logging
                 loss_tensor = torch.tensor([avg_train_loss_rank, avg_recon_loss_rank, avg_l1_loss_rank], device=self.device)
                 dist.all_reduce(loss_tensor, op=dist.ReduceOp.AVG)
                 avg_train_loss, avg_recon_loss, avg_l1_loss = loss_tensor.cpu().tolist()
            else:
                 avg_train_loss = avg_train_loss_rank
                 avg_recon_loss = avg_recon_loss_rank
                 avg_l1_loss = avg_l1_loss_rank

            # Log epoch-level metrics only on rank 0
            if self.rank == 0:
                log.info(f"Epoch {epoch+1} Train Complete: Avg Loss={avg_train_loss:.4f}, Avg Recon Loss={avg_recon_loss:.4f}, Avg L1 Loss={avg_l1_loss:.4f}")
                if self.wandb_run:
                    try:
                        wandb.log({
                            "epoch": epoch + 1,
                            "train/loss": avg_train_loss,
                            "train/recon_loss": avg_recon_loss,
                            "train/l1_loss": avg_l1_loss,
                            "learning_rate": self.optimizer.param_groups[0]['lr'] # Log LR
                        }, step=epoch+1)
                    except Exception as e:
                         log.error(f"Rank 0: Failed to log training metrics to WandB: {e}")
        elif self.rank == 0:
            log.warning(f"Epoch {epoch+1} Train: No batches processed on rank 0.")

    def _validate_epoch(self, epoch: int) -> float:
        """Runs one validation epoch with DDP, AMP support, tqdm, and returns *global* average loss."""
        self.model.eval() # Set model to evaluation mode

        # Set epoch for distributed sampler if used for validation
        if self.is_distributed and self.val_sampler:
             self.val_sampler.set_epoch(epoch) # Can use epoch or 0

        total_val_loss_rank = 0.0
        total_recon_loss_rank = 0.0
        total_l1_loss_rank = 0.0
        processed_batches_rank = 0 # Batches processed on this rank

        # Wrap the data loader with tqdm (only show on rank 0)
        val_pbar = tqdm(self.val_loader, desc=f"Epoch {epoch+1}/{self.num_epochs} [Val]",
                        leave=False, disable=self.rank != 0)

        with torch.no_grad(): # Disable gradient calculations
            for batch_data in val_pbar:
                batch_processed = self._process_batch(batch_data)
                target_batch = batch_processed

                # Fix 3: Use torch.amp.autocast with device_type as the first arg
                with torch.amp.autocast(device_type=self.device.type, enabled=self.amp_enabled):
                    # DDP forward pass (model is already wrapped)
                    # Note: In eval mode with no_grad, DDP doesn't need sync, but using the wrapped model is correct.
                    recon, encoded = self.model(batch_processed)
                    recon_loss = self.criterion(recon, target_batch)
                    l1_loss = self.l1 * torch.norm(encoded, p=1, dim=-1).mean()
                    loss = recon_loss + l1_loss

                # Update running losses for this rank
                total_val_loss_rank += loss.item()
                total_recon_loss_rank += recon_loss.item()
                total_l1_loss_rank += l1_loss.item()
                processed_batches_rank += 1

                # Update tqdm progress bar postfix (only shown on rank 0)
                if self.rank == 0:
                     current_loss = loss.item()
                     val_pbar.set_postfix(loss=f"{current_loss:.4f}")

        # --- Aggregate and Log Validation Metrics ---
        if processed_batches_rank > 0:
             # Pack metrics for aggregation
             metrics_tensor = torch.tensor([
                 total_val_loss_rank, total_recon_loss_rank, total_l1_loss_rank, processed_batches_rank
             ], device=self.device)

             if self.is_distributed:
                 # Sum metrics across all ranks
                 dist.all_reduce(metrics_tensor, op=dist.ReduceOp.SUM)

             # Unpack summed metrics (now identical on all ranks)
             total_val_loss_global = metrics_tensor[0].item()
             total_recon_loss_global = metrics_tensor[1].item()
             total_l1_loss_global = metrics_tensor[2].item()
             total_batches_global = metrics_tensor[3].item()

             # Calculate global averages
             avg_val_loss = total_val_loss_global / total_batches_global if total_batches_global > 0 else float('inf')
             avg_recon_loss = total_recon_loss_global / total_batches_global if total_batches_global > 0 else float('inf')
             avg_l1_loss = total_l1_loss_global / total_batches_global if total_batches_global > 0 else float('inf')

             # Log epoch-level metrics only on rank 0
             if self.rank == 0:
                 log.info(f"Epoch {epoch+1} Val Complete: Avg Loss={avg_val_loss:.4f}, Avg Recon Loss={avg_recon_loss:.4f}, Avg L1 Loss={avg_l1_loss:.4f}")
                 if self.wandb_run:
                      try:
                          wandb.log({
                              "epoch": epoch + 1,
                              "val/loss": avg_val_loss,
                              "val/recon_loss": avg_recon_loss,
                              "val/l1_loss": avg_l1_loss
                          }, step=epoch+1)
                      except Exception as e:
                           log.error(f"Rank 0: Failed to log validation metrics to WandB: {e}")

             return avg_val_loss # Return the global average loss

        elif self.rank == 0:
             log.warning(f"Epoch {epoch+1} Val: No batches processed on rank 0.")
             return float('inf') # Return infinity if no batches processed
        else:
             # Need to return something consistent on other ranks if rank 0 had no batches
             # Ideally, sync the result from rank 0, but returning inf is simpler if rank 0 handles saving best model
             return float('inf')


    def run(self):
        """Main execution method for the SAETrainer with DDP support."""
        try:
            # setup_wandb is rank-aware
            self._setup_wandb()
            # setup needs to complete on all ranks
            self._setup()
        except Exception as e:
             log.exception(f"Rank {self.rank}: Setup failed. Aborting run.")
             # Optional: Clean up distributed group? Depends on where it's initialized.
             return

        if self.rank == 0:
            log.info(f"Starting SAE training for {self.num_epochs} epochs...")
        start_time = time.time()

        # Wrap the epoch loop with tqdm (only show on rank 0)
        epoch_pbar = tqdm(range(self.num_epochs), desc="Overall Training Progress", disable=self.rank != 0)

        for epoch in epoch_pbar:
            epoch_start_time = time.time()
            # Update epoch pbar description (only relevant on rank 0)
            if self.rank == 0:
                epoch_pbar.set_description(f"Epoch {epoch+1}/{self.num_epochs}")

            self._train_epoch(epoch)

            # Perform validation periodically
            current_val_loss = float('inf')
            if (epoch + 1) % self.validate_every_n_epochs == 0:
                 current_val_loss = self._validate_epoch(epoch) # Returns global avg loss

                 # --- Checkpointing Logic (Rank 0 Only) ---
                 if self.rank == 0:
                     if current_val_loss < self.best_val_loss:
                         self.best_val_loss = current_val_loss
                         log.info(f"*** Rank 0: New best validation loss: {self.best_val_loss:.4f}. Saving checkpoint... ***")
                         try:
                             # _save_checkpoint is rank-aware
                             self._save_checkpoint(
                                 filename="best_model.pth",
                                 epoch=epoch + 1,
                                 val_loss=self.best_val_loss # Save best global loss
                             )
                         except Exception as e:
                              log.exception("Rank 0: Failed to save best checkpoint.")

            # Barrier to ensure all ranks finish epoch (esp. validation) before logging time
            if self.is_distributed:
                dist.barrier()

            # --- Log Epoch Timing (Rank 0 Only) ---
            if self.rank == 0:
                 epoch_duration = time.time() - epoch_start_time
                 log.info(f"--- Epoch {epoch+1} finished in {epoch_duration:.2f}s (Best Val Loss: {self.best_val_loss:.4f}) ---")
                 # Optionally update the main tqdm postfix
                 epoch_pbar.set_postfix(best_val_loss=f"{self.best_val_loss:.4f}")


        # --- Final Logging & Saving (Rank 0 Only) ---
        if self.rank == 0:
            total_duration = time.time() - start_time
            log.info(f"Training finished in {total_duration:.2f}s.")

            try:
                log.info("Saving final model checkpoint...")
                # _save_checkpoint is rank-aware
                self._save_checkpoint(filename="sae_final.pth", is_final=True, epoch=self.num_epochs)
            except Exception as e:
                log.exception("Failed to save final checkpoint.")

            # Optional: Final WandB cleanup
            if self.wandb_run:
                 wandb.finish()

        # Barrier ensures all ranks wait until rank 0 finishes saving/logging
        if self.is_distributed:
            dist.barrier()