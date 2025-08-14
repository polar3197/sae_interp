import hydra
from omegaconf import DictConfig, OmegaConf
import logging
import wandb
import os
# Import trainers to ensure registry or discovery works if not using explicit targets
# Not strictly necessary if using full _target_ paths in config
# from trainers import sae_trainer, probe_trainer

# Setup logging
log = logging.getLogger(__name__)

@hydra.main(config_path="../config", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    """
    Main entry point for training, configured by Hydra.

    Instantiates the appropriate trainer based on the configuration
    and runs the training process.
    """
    # Print the resolved configuration
    print("--- Configuration ---")
    print(OmegaConf.to_yaml(cfg))
    print("---------------------")
    print(f"Hydra working directory: {os.getcwd()}")
    print(f"Original working directory: {hydra.utils.get_original_cwd()}")
    print("---------------------")

    # Basic logging setup (could be more sophisticated)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(name)s: %(message)s')

    trainer = None # Initialize trainer variable
    try:
        # Instantiate the trainer specified in the configuration
        # Pass the full config (`cfg=cfg`) to the trainer's __init__
        # _recursive_=False prevents Hydra from auto-instantiating nested configs like
        # cfg.model, cfg.data if the trainer handles instantiation itself.
        log.info(f"Instantiating trainer: {cfg.trainer._target_}")
        trainer = hydra.utils.instantiate(cfg.trainer, cfg=cfg, _recursive_=False)
        log.info("Trainer instantiated successfully.")

        # Run the training process
        log.info("Starting trainer run...")
        trainer.run()
        log.info("Trainer run finished.")

    except Exception as e:
        log.exception("An error occurred during trainer instantiation or execution:")
        # Optionally finish wandb run with failure status if it was initialized
        if hasattr(trainer, 'wandb_run') and trainer.wandb_run:
            log.error("Finishing WandB run with failure status due to error.")
            wandb.finish(exit_code=1)
        # Re-raise the exception after logging and potentially finishing wandb
        raise e # Or handle differently, e.g., exit(1)

    finally:
        # --- WandB Finalization ---
        # Ensure WandB run finishes cleanly, regardless of trainer success/failure
        if wandb.run is not None:
            log.info("Finishing WandB run...")
            # Let wandb.finish handle the exit code based on exceptions
            wandb.finish()
            log.info("WandB run finished.")
        # Potentially other cleanup code here


if __name__ == "__main__":
    # Setup PYTHONPATH if necessary, e.g., export PYTHONPATH=$PYTHONPATH:$(pwd)
    # Make sure the 'src' directory is importable
    main() 