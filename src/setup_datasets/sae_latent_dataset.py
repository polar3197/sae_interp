import logging
import os
import sys
from pathlib import Path

import h5py
import hdf5plugin
import hydra
import numpy as np
import torch
from omegaconf import OmegaConf
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
log = logging.getLogger(__name__)

# --- Configuration (Global Variables) ---
INPUT_HDF5_PATH = Path("data/ve_latent_fairface.hdf5") # Path to the input HDF5 file (containing vision encoder 'encoded' dataset, 'labels', 'original_indices').
OUTPUT_RAW_DATA_HDF5_PATH = Path("data/sae_latent_fairface.hdf5") # Path for the output file containing raw token latents.
OUTPUT_AGG_DATA_HDF5_PATH = Path("data/agg_sae_latent_fairface.hdf5") # Path for the output file containing aggregated latents.
SAE_CHECKPOINT_PATH = Path("checkpoints/sae_final.pth")      # Path to the SAE model checkpoint (`.pth`). # TODO: Update with actual path
BATCH_SIZE = 128                  # Batch size (number of *input samples*, e.g., images, per iteration).
RAW_DATA_STORAGE_DTYPE = np.float16 # Dtype for raw SAE latents.
AGG_DATA_STORAGE_DTYPE = np.float16 # Dtype for mean/max aggregated latents.
LABEL_STORAGE_DTYPE = np.uint8    # Expected label type. # This is informational, labels are copied directly
COMPRESSION_LEVEL = 3             # Zstd compression level (1-22)


def main():
    """
    Main function to generate SAE encoded datasets (raw and aggregated)
    into two separate HDF5 files.
    Uses global variables for configuration.
    """

    # --- Input Validation ---
    if not INPUT_HDF5_PATH.is_file():
        log.error(f"Input HDF5 file not found: {INPUT_HDF5_PATH}")
        sys.exit(1)
    if not SAE_CHECKPOINT_PATH.is_file():
        log.error(f"SAE Checkpoint file not found: {SAE_CHECKPOINT_PATH}")
        sys.exit(1)

    # Create output directories if needed
    OUTPUT_RAW_DATA_HDF5_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_AGG_DATA_HDF5_PATH.parent.mkdir(parents=True, exist_ok=True)

    log.info(f"Input HDF5: {INPUT_HDF5_PATH}")
    log.info(f"Output Raw HDF5: {OUTPUT_RAW_DATA_HDF5_PATH}")
    log.info(f"Output Aggregated HDF5: {OUTPUT_AGG_DATA_HDF5_PATH}")
    log.info(f"SAE Checkpoint: {SAE_CHECKPOINT_PATH}")
    log.info(f"Batch Size: {BATCH_SIZE}")
    log.info(f"Raw Data Dtype: {RAW_DATA_STORAGE_DTYPE}")
    log.info(f"Aggregated Data Dtype: {AGG_DATA_STORAGE_DTYPE}")
    log.info(f"Compression Level: {COMPRESSION_LEVEL}")

    # --- Define Compression ---
    zstd_compression = hdf5plugin.Zstd(clevel=COMPRESSION_LEVEL)

    # --- Determine Device ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Using device: {device}")

    # --- Load SAE Model ---
    log.info("Loading SAE model from checkpoint...")
    try:
        checkpoint = torch.load(SAE_CHECKPOINT_PATH, map_location='cpu')
        if 'config' not in checkpoint:
             log.error("Checkpoint does not contain 'config' key for Hydra instantiation.")
             sys.exit(1)
        if 'model_state_dict' not in checkpoint:
             log.error("Checkpoint does not contain 'model_state_dict' key.")
             sys.exit(1)

        cfg = OmegaConf.create(checkpoint['config'])
        # Ensure hydra.utils is available or adjust instantiation as needed
        model = hydra.utils.instantiate(cfg.model)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()
        model_dtype = next(model.parameters()).dtype # Get model's expected dtype

        # Determine latent dimension (d_latent)
        d_latent = None
        if hasattr(model, 'd_sae'):
            d_latent = model.d_sae
        elif hasattr(model, 'cfg') and hasattr(model.cfg, 'd_sae'):
             d_latent = model.cfg.d_sae
        elif hasattr(model, 'encoder') and hasattr(model.encoder, 'out_features'):
             d_latent = model.encoder.out_features
        elif hasattr(model, 'W_enc'):
            d_latent = model.W_enc.shape[0]

        if d_latent is None:
            log.warning("Could not automatically determine latent dimension (D_latent).")
            # Attempt dummy inference
            dummy_embedding_dim = 768 # Default guess
            try:
                 with h5py.File(INPUT_HDF5_PATH, 'r') as f_in_dummy:
                      if 'training' in f_in_dummy and 'encoded' in f_in_dummy['training']:
                           dummy_embedding_dim = f_in_dummy['training']['encoded'].shape[2]
                           log.info(f"Read embedding_dim={dummy_embedding_dim} from input file.")
                      elif 'validation' in f_in_dummy and 'encoded' in f_in_dummy['validation']:
                           dummy_embedding_dim = f_in_dummy['validation']['encoded'].shape[2]
                           log.info(f"Read embedding_dim={dummy_embedding_dim} from input file.")
                      else:
                          log.warning(f"Could not read embedding_dim from input file, using default {dummy_embedding_dim}")
            except Exception as e:
                 log.warning(f"Could not read embedding_dim from input file ({e}), using default {dummy_embedding_dim}")

            log.info(f"Attempting dummy inference to find D_latent using embedding_dim={dummy_embedding_dim}...")
            dummy_input = torch.randn(2, dummy_embedding_dim, device=device, dtype=model_dtype)
            try:
                with torch.no_grad():
                    if hasattr(model, 'encode') and callable(model.encode):
                        dummy_output = model.encode(dummy_input)
                    else:
                        dummy_output = model(dummy_input)

                    if isinstance(dummy_output, tuple):
                        dummy_output = dummy_output[1] # Assume second element is latent
                        log.info("SAE output is a tuple, assuming second element is latent.")

                    if dummy_output.ndim == 2:
                        d_latent = dummy_output.shape[-1]
                        log.info(f"Inferred D_latent = {d_latent} from dummy output shape {dummy_output.shape}.")
                    else:
                        log.error(f"Dummy output has unexpected shape {dummy_output.shape}. Expected (batch, d_latent).")
                        sys.exit(1)
            except Exception as e:
                 log.error(f"Dummy inference failed: {e}")
                 log.error("Cannot determine latent dimension.")
                 sys.exit(1)
        else:
             log.info(f"Determined latent dimension D_latent = {d_latent}")

    except Exception as e:
        log.exception(f"Failed to load model: {e}")
        sys.exit(1)

    # --- Process Data ---
    try:
        # Use nested with statements for the three files
        with h5py.File(INPUT_HDF5_PATH, 'r') as f_in, \
             h5py.File(OUTPUT_RAW_DATA_HDF5_PATH, 'w') as f_out_raw, \
             h5py.File(OUTPUT_AGG_DATA_HDF5_PATH, 'w') as f_out_agg:

            log.info("Input and output HDF5 files opened.")

            # --- Copy top-level attributes ---
            log.info("Copying top-level attributes to both output files...")
            for key, value in f_in.attrs.items():
                try:
                    f_out_raw.attrs[key] = value
                    f_out_agg.attrs[key] = value
                except Exception as attr_err:
                    log.warning(f"Could not copy attribute '{key}': {attr_err}")
            log.info("Top-level attributes copied.")

            for split in ['training', 'validation']:
                log.info(f"--- Processing split: {split} ---")
                if split not in f_in:
                    log.warning(f"Split '{split}' not found in input file. Skipping.")
                    continue

                input_group = f_in[split]

                # Create output groups in both files
                output_group_raw = f_out_raw.create_group(split)
                output_group_agg = f_out_agg.create_group(split)

                 # Copy attributes from input group to both output groups
                log.info(f"Copying attributes for group '{split}'...")
                for key, value in input_group.attrs.items():
                    try:
                        output_group_raw.attrs[key] = value
                        output_group_agg.attrs[key] = value
                    except Exception as attr_err:
                        log.warning(f"Could not copy group attribute '{key}' for split '{split}': {attr_err}")

                if 'encoded' not in input_group:
                    log.warning(f"'encoded' dataset not found in input split '{split}'. Cannot generate latents. Skipping split processing.")
                    # Copy labels/indices if they exist even if no data
                    if 'labels' in input_group:
                        log.info(f"Copying labels group for empty data split '{split}'...")
                        try:
                            f_out_raw.copy(input_group['labels'], output_group_raw, name='labels')
                            f_out_agg.copy(input_group['labels'], output_group_agg, name='labels')
                        except Exception as copy_err:
                             log.warning(f"Could not copy labels for empty split '{split}': {copy_err}")
                    if 'original_indices' in input_group:
                         log.info(f"Copying original_indices dataset for empty data split '{split}'...")
                         try:
                             f_out_raw.copy(input_group['original_indices'], output_group_raw, name='original_indices')
                             f_out_agg.copy(input_group['original_indices'], output_group_agg, name='original_indices')
                         except Exception as copy_err:
                              log.warning(f"Could not copy original_indices for empty split '{split}': {copy_err}")
                    continue # Skip to next split

                # Get input details
                input_encoded_dset = input_group['encoded']
                n_samples, sequence_length, embedding_dim = input_encoded_dset.shape
                log.info(f"Input '{split}/encoded': {n_samples} samples, seq_len={sequence_length}, embed_dim={embedding_dim}.")

                has_labels = 'labels' in input_group
                has_original_indices = 'original_indices' in input_group

                # Handle n_samples == 0 case
                if n_samples == 0:
                    log.warning(f"Split '{split}' has 0 samples in 'encoded' dataset. Skipping processing.")
                    # Copy labels/indices if they exist
                    if has_labels:
                         log.info(f"Copying labels group for 0-sample split '{split}'...")
                         try:
                            f_out_raw.copy(input_group['labels'], output_group_raw, name='labels')
                            f_out_agg.copy(input_group['labels'], output_group_agg, name='labels')
                         except Exception as copy_err:
                             log.warning(f"Could not copy labels for 0-sample split '{split}': {copy_err}")
                    if has_original_indices:
                         log.info(f"Copying original_indices dataset for 0-sample split '{split}'...")
                         try:
                            f_out_raw.copy(input_group['original_indices'], output_group_raw, name='original_indices')
                            f_out_agg.copy(input_group['original_indices'], output_group_agg, name='original_indices')
                         except Exception as copy_err:
                             log.warning(f"Could not copy original_indices for 0-sample split '{split}': {copy_err}")
                    continue # Skip to next split

                # --- Define parameters for cropping ---
                n_tokens_per_sample = sequence_length - 1
                if n_tokens_per_sample <= 0:
                    log.error(f"Sequence length ({sequence_length}) is too short to crop (<=1). Skipping split '{split}'.")
                    continue
                log.info(f"Using {n_tokens_per_sample} tokens per sample for SAE input.")

                # --- Create Output Datasets in Respective Files ---
                raw_data_shape = (n_samples, n_tokens_per_sample, d_latent)
                aggregated_shape = (n_samples, d_latent)
                raw_data_chunks = (min(BATCH_SIZE, n_samples), n_tokens_per_sample, d_latent)
                agg_chunks = (min(BATCH_SIZE * 4, n_samples), d_latent) # Adjust chunk factor as needed

                log.info(f"Creating dataset '{split}/data' in {OUTPUT_RAW_DATA_HDF5_PATH.name} "
                         f"with shape={raw_data_shape}, dtype={RAW_DATA_STORAGE_DTYPE}, chunks={raw_data_chunks}")
                output_raw_data_dset = output_group_raw.create_dataset(
                    'data', # Renamed from 'latent_activations'
                    shape=raw_data_shape,
                    dtype=RAW_DATA_STORAGE_DTYPE,
                    chunks=raw_data_chunks,
                    compression=zstd_compression,
                )

                log.info(f"Creating group '{split}/data' in {OUTPUT_AGG_DATA_HDF5_PATH.name}")
                data_agg_group = output_group_agg.create_group('data') # New group for aggregated data

                log.info(f"Creating dataset '{split}/data/mean' in {OUTPUT_AGG_DATA_HDF5_PATH.name} "
                         f"with shape={aggregated_shape}, dtype={AGG_DATA_STORAGE_DTYPE}, chunks={agg_chunks}")
                output_mean_dset = data_agg_group.create_dataset(
                    'mean', # Renamed and moved
                    shape=aggregated_shape,
                    dtype=AGG_DATA_STORAGE_DTYPE,
                    chunks=agg_chunks,
                    compression=zstd_compression,
                )

                log.info(f"Creating dataset '{split}/data/max' in {OUTPUT_AGG_DATA_HDF5_PATH.name} "
                         f"with shape={aggregated_shape}, dtype={AGG_DATA_STORAGE_DTYPE}, chunks={agg_chunks}")
                output_max_dset = data_agg_group.create_dataset(
                    'max', # Renamed and moved
                    shape=aggregated_shape,
                    dtype=AGG_DATA_STORAGE_DTYPE,
                    chunks=agg_chunks,
                    compression=zstd_compression,
                )

                # --- Copy labels (if they exist) ---
                if has_labels:
                    log.info(f"Copying labels group for split '{split}' to both output files...")
                    try:
                        f_out_raw.copy(input_group['labels'], output_group_raw, name='labels')
                        f_out_agg.copy(input_group['labels'], output_group_agg, name='labels')
                        log.info("Labels copied.")
                    except Exception as copy_err:
                        log.warning(f"Could not copy labels group for split '{split}': {copy_err}")
                else:
                    log.warning(f"No 'labels' group found in input split '{split}' to copy.")

                # --- Copy original_indices (if they exist) ---
                if has_original_indices:
                    log.info(f"Copying 'original_indices' dataset for split '{split}' to both output files...")
                    try:
                        f_out_raw.copy(input_group['original_indices'], output_group_raw, name='original_indices')
                        f_out_agg.copy(input_group['original_indices'], output_group_agg, name='original_indices')
                        log.info("'original_indices' copied.")
                    except Exception as copy_err:
                        log.warning(f"Could not copy original_indices dataset for split '{split}': {copy_err}")
                else:
                    log.warning(f"No 'original_indices' dataset found in input split '{split}' to copy.")

                # --- Process the split in batches ---
                log.info(f"Processing and aggregating split '{split}' with batch_size={BATCH_SIZE}...")
                with torch.no_grad():
                    for i in tqdm(range(0, n_samples, BATCH_SIZE), desc=f"Processing {split} batches"):
                        start_idx = i
                        end_idx = min(i + BATCH_SIZE, n_samples)
                        current_batch_size = end_idx - start_idx
                        if current_batch_size == 0: continue

                        # Read, Crop, Flatten
                        input_batch_np = input_encoded_dset[start_idx:end_idx]
                        cropped_batch_np = input_batch_np[:, 1:, :]
                        flattened_batch_np = cropped_batch_np.reshape(-1, embedding_dim)
                        n_tokens_in_batch = flattened_batch_np.shape[0]
                        if n_tokens_in_batch == 0: continue

                        # Prepare & Encode
                        flattened_batch_torch = torch.from_numpy(flattened_batch_np).to(device).to(model_dtype)
                        if hasattr(model, 'encode') and callable(model.encode):
                            latent_batch_torch = model.encode(flattened_batch_torch)
                        else:
                            latent_batch_torch = model(flattened_batch_torch)
                            if isinstance(latent_batch_torch, tuple):
                                latent_batch_torch = latent_batch_torch[1] # Assuming [1] is latents

                        # Prepare for Storage/Aggregation
                        latent_batch_np = latent_batch_torch.cpu().numpy()

                        # Reshape Raw Data (B, S-1, D_sae)
                        reshaped_raw_data_np = latent_batch_np.reshape(
                            current_batch_size, n_tokens_per_sample, d_latent
                        ) # Keep higher precision for agg

                        # Write Raw Data (converting dtype just before write)
                        try:
                            output_raw_data_dset[start_idx:end_idx, :, :] = reshaped_raw_data_np.astype(RAW_DATA_STORAGE_DTYPE)
                        except Exception as write_err:
                            log.error(f"Error writing raw data batch {start_idx}:{end_idx}: {write_err}")
                            break # Stop processing this split on write error

                        # Aggregate (use float32 for stability)
                        agg_input_f32 = reshaped_raw_data_np.astype(np.float32)
                        mean_agg_batch = np.mean(agg_input_f32, axis=1).astype(AGG_DATA_STORAGE_DTYPE)
                        max_agg_batch = np.max(agg_input_f32, axis=1).astype(AGG_DATA_STORAGE_DTYPE)

                        # Write Aggregated Data
                        try:
                            output_mean_dset[start_idx:end_idx] = mean_agg_batch
                            output_max_dset[start_idx:end_idx] = max_agg_batch
                        except Exception as write_err:
                             log.error(f"Error writing aggregated data batch {start_idx}:{end_idx}: {write_err}")
                             break # Stop processing this split on write error

                log.info(f"Finished processing split {split}.")

        log.info(f"Successfully created SAE datasets:")
        log.info(f"  Raw token data: {OUTPUT_RAW_DATA_HDF5_PATH}")
        log.info(f"  Aggregated data: {OUTPUT_AGG_DATA_HDF5_PATH}")

    except Exception as e:
        log.exception(f"An error occurred during processing: {e}")
        # Clean up potentially partially written output files
        files_to_remove = [OUTPUT_RAW_DATA_HDF5_PATH, OUTPUT_AGG_DATA_HDF5_PATH]
        for file_path in files_to_remove:
            if file_path.exists():
                log.warning(f"Attempting to remove partially written file: {file_path}")
                try:
                    os.remove(file_path)
                except OSError as rm_err:
                    log.error(f"Failed to remove partial file {file_path}: {rm_err}")
        sys.exit(1)


if __name__ == "__main__":
    # No command-line arguments, just run main which uses globals
    main()