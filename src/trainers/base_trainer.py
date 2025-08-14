import torch
from omegaconf import DictConfig, OmegaConf
import wandb
import os
import logging
import torch.distributed as dist # Import distributed

log = logging.getLogger(__name__)

class BaseTrainer:
    def __init__(self, cfg: DictConfig, rank: int = 0, local_rank: int = 0, world_size: int = 1):
        self.cfg = cfg
        self.rank = rank
        self.local_rank = local_rank
        self.world_size = world_size
        self.is_distributed = world_size > 1
        self.device = self._setup_device()
        self.wandb_run = None # To store the initialized wandb run

    def _setup_device(self) -> torch.device:
        """Sets up the device based on config, availability, and DDP rank."""
        if self.is_distributed:
             # DDP requires CUDA, device is set by local_rank
             if not torch.cuda.is_available():
                 log.error("Distributed training requires CUDA.")
                 raise RuntimeError("Distributed training requires CUDA.")
             device = torch.device(f"cuda:{self.local_rank}")
             # torch.cuda.set_device(device) # Main script should handle this
             log.info(f"Rank {self.rank}/{self.world_size} using device: {device}")
        else:
             # Single process logic
             if self.cfg.device == "auto":
                 device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
             else:
                 device = torch.device(self.cfg.device)
             log.info(f"Using device: {device}")
        return device

    def _setup_wandb(self) -> None:
        """Initializes Weights & Biases run, only on rank 0."""
        if self.rank == 0: # Only initialize on the main process
             if self.wandb_run is None:
                 log.info(f"Rank {self.rank}: Initializing WandB run... Project: {self.cfg.wandb.project}")
                 try:
                     self.wandb_run = wandb.init(
                         project=self.cfg.wandb.project,
                         entity=self.cfg.wandb.get("entity"),
                         config=OmegaConf.to_container(self.cfg, resolve=True, throw_on_missing=True),
                         # name=hydra.core.hydra_config.HydraConfig.get().job.name # Optional
                     )
                     log.info(f"Rank {self.rank}: WandB Run Name: {self.wandb_run.name}, ID: {self.wandb_run.id}")
                 except Exception as e:
                      log.exception(f"Rank {self.rank}: Failed to initialize WandB.")
                      self.wandb_run = None # Ensure it's None if init fails
             else:
                 log.warning(f"Rank {self.rank}: WandB already initialized.")
        else:
             log.debug(f"Rank {self.rank}: Skipping WandB initialization.")


    def _save_checkpoint(self, filename: str = "checkpoint.pth", **kwargs) -> None:
        """Saves model, optimizer, and scaler state, only on rank 0."""
        if self.rank == 0: # Only save on the main process
             log.info(f"Rank {self.rank}: Preparing to save checkpoint '{filename}'...")
             # Ensure model is on CPU before saving to avoid GPU mapping issues on load
             # Need to handle potential DDP wrapping here before accessing state_dict
             model_to_save = self.model.module if hasattr(self.model, 'module') else self.model
             model_to_save.cpu() # Move the underlying model to CPU

             state = {
                 'model_state_dict': model_to_save.state_dict(),
                 'optimizer_state_dict': self.optimizer.state_dict() if hasattr(self, 'optimizer') and self.optimizer is not None else None,
                 # Save scaler state if it exists (used by AMP)
                 'scaler_state_dict': self.scaler.state_dict() if hasattr(self, 'scaler') and self.scaler is not None else None,
                 'config': OmegaConf.to_container(self.cfg, resolve=True),
                 **kwargs # Allow saving extra info like epoch, val_loss
             }
             save_path = os.path.join(os.getcwd(), filename) # Save in Hydra output dir
             try:
                 torch.save(state, save_path)
                 log.info(f"Rank {self.rank}: Checkpoint saved to {save_path}")

                 # Log as artifact if enabled and wandb run exists
                 if self.cfg.wandb.log_model and self.wandb_run:
                      artifact_name = f"{self.wandb_run.id}-{filename.split('.')[0]}"
                      artifact = wandb.Artifact(name=artifact_name, type="model")
                      artifact.add_file(save_path)
                      self.wandb_run.log_artifact(artifact)
                      log.info(f"Rank {self.rank}: Checkpoint logged as WandB artifact: {artifact.name}")

             except Exception as e:
                  log.exception(f"Rank {self.rank}: Failed to save checkpoint to {save_path}")
             finally:
                  # Move model back to original device regardless of success/failure
                  # If DDP wrapped, the original self.model still points to the DDP wrapper
                  self.model.to(self.device)
        else:
             log.debug(f"Rank {self.rank}: Skipping checkpoint save.")

    # Placeholder for the main run method, to be implemented by subclasses
    def run(self):
        raise NotImplementedError("Subclasses must implement the 'run' method.")

    # Optional: _load_checkpoint method if needed
