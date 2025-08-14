import torch
import h5py
import hdf5plugin # For Zstandard compression
import numpy as np
import pandas as pd # For stratification
import random # For shuffling indices
from tqdm.auto import tqdm
from PIL import Image
import sys
import os
from typing import Dict, List, Tuple, Optional
from functools import partial # For cleaner collate_fn binding

# Set environment variable to suppress the tokenizer parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Ensure the src directory is in the Python path
script_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(script_dir, '..', 'src')
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

# Now import from src
try:
    from src.deconstructed_florence import DeconstructedFlorence2, FlorenceVisionEncoder
    from transformers import AutoProcessor
    from datasets import load_dataset, Dataset, Features, Value, Image as HFImage, ClassLabel
except ImportError as e:
    print(f"Error importing necessary modules: {e}", file=sys.stderr)
    print(f"Attempted to import from: {src_dir}", file=sys.stderr)
    print("Ensure 'src', 'transformers', 'datasets', 'pandas', 'hdf5plugin' are installed.", file=sys.stderr)
    sys.exit(1)

# --- Configuration ---
DATASET_NAME = 'HuggingFaceM4/FairFace'
DATASET_CONFIG = '0.25' # Use the smaller 0.25 version of FairFace
IMAGE_COLUMN = 'image'
LABEL_COLUMNS_TO_STORE = ['age', 'gender', 'race'] # Specific labels for FairFace to keep
MODEL_ID = "microsoft/Florence-2-base"
HDF5_OUTPUT_PATH = "data/ve_latent_fairface.hdf5" # Specific output name for subset
BATCH_SIZE = 64          # Adjust based on GPU memory
FORCE_CPU = False        # Set to True to force CPU usage
NUM_WORKERS = 4          # Number of DataLoader workers
ZSTD_CLEVEL = 3          # Zstandard Compression level (1-22, higher is smaller but slower)

# --- Subsetting Configuration ---
STRATIFY_COLUMNS = ['race', 'gender', 'age'] # Columns to stratify on (e.g., ['race'], ['race', 'gender'])
TARGET_SAMPLES_PER_STRATUM_TRAIN = 100 # Desired train samples per stratum combination
TARGET_SAMPLES_PER_STRATUM_VAL = 25   # Desired validation samples per stratum combination
RANDOM_SEED = 42         # For reproducible sampling

# --- Setup Device and Dtype ---
if FORCE_CPU:
    DEVICE = 'cpu'
else:
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Compute Dtype (Model runs with this)
COMPUTE_DTYPE = torch.bfloat16 if DEVICE == 'cuda' and torch.cuda.is_bf16_supported() else torch.float32
# Storage Dtypes (HDF5 stores these - float16 for encoded, uint8 for labels)
FLOAT_STORAGE_DTYPE = np.float16
LABEL_STORAGE_DTYPE = np.uint8 # Assuming < 256 classes for age, gender, race in FairFace
print(f"Using device: {DEVICE}, compute dtype: {COMPUTE_DTYPE}")
print(f"Storage dtypes: float={FLOAT_STORAGE_DTYPE}, label={LABEL_STORAGE_DTYPE}")
print(f"Using Zstandard compression level: {ZSTD_CLEVEL}")
PIN_MEMORY = (DEVICE == 'cuda')
random.seed(RANDOM_SEED) # Set random seed for shuffling

# --- Helper Functions ---

def check_columns(dataset_features: Features, image_col: str, label_cols: List[str], stratify_cols: List[str]):
    """Validate required image, label, and stratification columns exist."""
    required_cols = set([image_col] + label_cols + stratify_cols)
    available_cols = set(dataset_features.keys())
    missing_cols = required_cols - available_cols

    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}. Available: {list(available_cols)}")

    if not isinstance(dataset_features[image_col], HFImage):
         print(f"Warning: Image column '{image_col}' is not of type datasets.Image. Ensure it contains loadable PIL images.", file=sys.stderr)

    for label_col in label_cols + stratify_cols: # Check labels and stratify cols typing
        # FairFace uses ClassLabel, which is expected
        if not isinstance(dataset_features[label_col], ClassLabel):
             print(f"Warning: Column '{label_col}' is not of type datasets.ClassLabel. Ensure it's suitable for stratification/storage.", file=sys.stderr)


def create_stratified_subset(dataset: Dataset, stratify_cols: List[str], target_per_stratum: int, random_seed: int) -> Tuple[Dataset, List[int]]:
    """Creates a stratified subset of the dataset using pandas, returns subset and indices."""
    print(f"Creating stratified subset based on {stratify_cols}, target/stratum={target_per_stratum}...")
    if not stratify_cols:
        raise ValueError("Stratification columns must be specified.")

    try:
        df = dataset.to_pandas()
    except Exception as e:
         print(f"Error converting dataset split to Pandas DataFrame: {e}", file=sys.stderr)
         print("Ensure the dataset structure is compatible with Pandas conversion.", file=sys.stderr)
         raise

    # Ensure stratify columns are suitable type (often categorical/int from ClassLabel)
    for col in stratify_cols:
        if not pd.api.types.is_integer_dtype(df[col]) and not pd.api.types.is_categorical_dtype(df[col]):
            print(f"Warning: Stratification column '{col}' has dtype {df[col].dtype}. Converting to category for grouping.", file=sys.stderr)
            try:
                df[col] = df[col].astype('category')
            except Exception as e:
                 print(f"Failed to convert column '{col}' to category: {e}", file=sys.stderr)


    grouped = df.groupby(stratify_cols, observed=False) # observed=False needed for categories sometimes
    sampled_indices_list = [] # Renamed to avoid confusion with pandas index object

    print(f"  Found {grouped.ngroups} strata combinations.")
    num_sampled_total = 0
    # Use tqdm on grouped.groups.items() for progress bar with names if needed, or keep simple loop
    for name, group_df in tqdm(grouped, desc="Sampling strata", total=grouped.ngroups): # group is now group_df
        n_available = len(group_df)
        n_samples = min(target_per_stratum, n_available)
        if n_samples > 0:
            # Sample indices directly from the group's index
            sampled_group_indices = group_df.sample(n=n_samples, random_state=random_seed).index
            sampled_indices_list.extend(sampled_group_indices.tolist()) # Add indices to the list
            num_sampled_total += n_samples

    print(f"  Total samples selected before shuffling: {len(sampled_indices_list)}")

    if not sampled_indices_list:
         print("Warning: No samples were selected after stratification.", file=sys.stderr)
         # Return an empty dataset and empty list
         return dataset.select([]), []


    # Shuffle the final indices to mix strata for DataLoader
    # Make a copy before shuffling if you need the original order for some reason
    final_shuffled_indices = sampled_indices_list.copy()
    random.shuffle(final_shuffled_indices)

    # Select the subset from the original dataset using the shuffled list of indices
    subset_dataset = dataset.select(final_shuffled_indices)
    print(f"  Final subset size: {len(subset_dataset)}")
    # Return the subset AND the shuffled list of indices used to create it
    return subset_dataset, final_shuffled_indices


def get_shapes_and_dtypes(
    processor: AutoProcessor,
    vision_encoder: FlorenceVisionEncoder,
    sample_image: Image.Image,
    label_columns: List[str], # Only the labels to be stored
    device: str,
    compute_dtype: torch.dtype
) -> Tuple[Tuple, np.dtype, Dict[str, np.dtype]]: # Returns only encoded shape/dtype and label dtypes
    """Process one sample to determine shapes and storage dtypes for HDF5 (only encoded)."""
    dummy_text = "<DUMMY>" # Placeholder text for processor
    inputs = processor(text=dummy_text, images=sample_image, return_tensors="pt")
    pixel_values_sample = inputs['pixel_values'].to(device=device, dtype=compute_dtype)

    with torch.no_grad():
        encoded_features_sample = vision_encoder(pixel_values_sample)

    encoded_shape = encoded_features_sample.shape # Includes batch dim 1

    # Determine storage dtypes (float16 for encoded, uint8 for labels)
    encoded_storage_dtype = FLOAT_STORAGE_DTYPE
    label_storage_dtypes = {label_col: LABEL_STORAGE_DTYPE for label_col in label_columns}

    # Return shapes *without* the sample batch dimension
    encoded_shape_tpl = tuple(encoded_shape[1:])

    # Return only encoded shape/dtype and label dtypes
    return encoded_shape_tpl, encoded_storage_dtype, label_storage_dtypes

# Define the custom collate function for FairFace
def collate_batch(batch_list: List[Dict], processor: AutoProcessor, image_col: str, label_cols: List[str]) -> Dict:
    """
    Collates FairFace samples: processes images and packages multiple labels to store.
    """
    images = [item[image_col] for item in batch_list]
    # Create a dictionary of label lists FOR LABELS TO STORE
    labels_dict = {label_col: [item[label_col] for item in batch_list] for label_col in label_cols}

    dummy_texts = ["<IGNORE>" for _ in images]

    try:
        # Preprocess images
        inputs = processor(text=dummy_texts, images=images, return_tensors="pt")
        pixel_values = inputs['pixel_values']
    except Exception as e:
        # Handle potential errors during image loading/processing within a batch
        print(f"\nError in collate_fn processing images: {e}", file=sys.stderr)
        # Attempt to identify problematic items (optional, can be slow)
        # for i, item in enumerate(batch_list):
        #     try:
        #         processor(text="<IGNORE>", images=item[image_col], return_tensors="pt")
        #     except Exception as item_e:
        #         print(f"  -> Problematic item index {i}: {item}. Error: {item_e}", file=sys.stderr)

        # Return empty tensors or handle as appropriate
        # Returning empty signifies batch failure upstream
        return {'pixel_values': torch.empty(0), 'labels': {label_col: torch.empty(0, dtype=torch.long) for label_col in label_cols}}


    # Convert each label list (for stored labels) to a tensor
    labels_tensor_dict = {
        label_col: torch.tensor(labels, dtype=torch.long)
        for label_col, labels in labels_dict.items()
    }

    # Return pixel values and the dictionary of label tensors to be stored
    return {'pixel_values': pixel_values, 'labels': labels_tensor_dict}


def process_and_store_split(
    dataset_split: Dataset, # This will be the SUBSET
    hdf5_group: h5py.Group,
    processor: AutoProcessor,
    vision_encoder: FlorenceVisionEncoder,
    image_col: str,
    label_cols_to_store: List[str], # Only store these labels
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
    device: str,
    compute_dtype: torch.dtype,
    split_name: str
):
    """Processes a dataset SUBSET split and stores encoded features and labels in the HDF5 group."""
    print(f"\nProcessing split: {split_name}")
    current_offset = 0

    # Access HDF5 datasets (only encoded and labels)
    encoded_dset = hdf5_group['encoded']
    label_dsets = {label_col: hdf5_group[f'labels/{label_col}'] for label_col in label_cols_to_store}

    # --- Batch Processing ---
    # Use functools.partial for cleaner binding of arguments to collate_fn
    collate_fn_partial = partial(collate_batch, processor=processor, image_col=image_col, label_cols=label_cols_to_store)

    dataloader = torch.utils.data.DataLoader(
        dataset_split, # Use the subset dataset
        batch_size=batch_size,
        collate_fn=collate_fn_partial,
        num_workers=num_workers,
        pin_memory=pin_memory,
        # persistent_workers=True if num_workers > 0 else False # Can speed up if workers > 0
    )

    total_items = len(dataset_split) # Total items in the subset for tqdm
    with tqdm(total=total_items, desc=f"Processing {split_name}", unit="img") as pbar:
        for batch in dataloader:
            pixel_values = batch['pixel_values']
            labels_batch_dict = batch['labels'] # Dict of label tensors to store

            # Handle potential empty batches from collate_fn errors
            current_batch_size = pixel_values.shape[0]
            if current_batch_size == 0:
                 print("Warning: Skipping empty batch returned by collate_fn.", file=sys.stderr)
                 continue

            pixel_values = pixel_values.to(device=device, dtype=compute_dtype, non_blocking=pin_memory)

            with torch.no_grad():
                # Get only the final encoded features
                encoded_features = vision_encoder(pixel_values)

            # Write data directly to the pre-allocated slice
            end_offset = current_offset + current_batch_size
            if end_offset > total_items:
                print(f"Warning: Attempting to write beyond allocated size ({end_offset} > {total_items}). Trimming.", file=sys.stderr)
                end_offset = total_items
                current_batch_size = total_items - current_offset
                if current_batch_size <= 0: break # Stop if we somehow went over
                encoded_features = encoded_features[:current_batch_size]


            # Cast encoded features to float16 for storage
            encoded_dset[current_offset:end_offset] = encoded_features.cpu().to(torch.float16).numpy()

            # Write each label type to store, casting to uint8
            for label_name, label_tensor in labels_batch_dict.items():
                label_dset = label_dsets[label_name]
                # Ensure label tensor is also trimmed if batch was trimmed
                label_tensor_trimmed = label_tensor[:current_batch_size]
                label_dset[current_offset:end_offset] = label_tensor_trimmed.cpu().numpy().astype(LABEL_STORAGE_DTYPE)

            current_offset = end_offset
            pbar.update(current_batch_size)

    # Final check
    if current_offset != total_items:
         print(f"Warning: Final offset ({current_offset}) does not match expected subset size ({total_items}) for split '{split_name}'.", file=sys.stderr)
    else:
         print(f"Finished processing {split_name}. Total items processed: {current_offset}")


# --- Main Execution ---
if __name__ == "__main__":
    print("Starting Stratified FairFace Subset Curation (Memory Optimized, Zstd)...")

    # 1. Load Full Dataset & Get Sizes
    print(f"Loading full dataset: {DATASET_NAME} ({DATASET_CONFIG})")
    try:
        full_dataset = load_dataset(DATASET_NAME, DATASET_CONFIG)
    except Exception as e:
        print(f"Error loading dataset '{DATASET_NAME}': {e}", file=sys.stderr)
        sys.exit(1)

    # FairFace has 'train' and 'validation' splits
    if 'train' not in full_dataset or 'validation' not in full_dataset:
        raise ValueError(f"Dataset '{DATASET_NAME}' must contain 'train' and 'validation' splits.")

    original_train_dataset = full_dataset['train']
    original_val_dataset = full_dataset['validation']

    N_train_original = len(original_train_dataset)
    N_val_original = len(original_val_dataset)
    print(f"Original train samples: {N_train_original}, Original validation samples: {N_val_original}")

    # Validate columns on original dataset
    try:
         check_columns(original_train_dataset.features, IMAGE_COLUMN, LABEL_COLUMNS_TO_STORE, STRATIFY_COLUMNS)
         check_columns(original_val_dataset.features, IMAGE_COLUMN, LABEL_COLUMNS_TO_STORE, STRATIFY_COLUMNS)
    except ValueError as e:
         print(e, file=sys.stderr)
         sys.exit(1)

    # 2. Perform Stratified Subsetting - Capture indices
    print("\nCreating subsets...")
    train_subset, train_indices = create_stratified_subset(original_train_dataset, STRATIFY_COLUMNS, TARGET_SAMPLES_PER_STRATUM_TRAIN, RANDOM_SEED)
    val_subset, val_indices = create_stratified_subset(original_val_dataset, STRATIFY_COLUMNS, TARGET_SAMPLES_PER_STRATUM_VAL, RANDOM_SEED)

    N_train_subset = len(train_subset)
    N_val_subset = len(val_subset)
    print(f"\nSubset train samples: {N_train_subset}, Subset validation samples: {N_val_subset}")

    if N_train_subset == 0 and N_val_subset == 0:
        print("Both subsets are empty. Exiting.")
        sys.exit(0)


    # 3. Load Model and Processor
    print(f"\nLoading model: {MODEL_ID}")
    try:
        # Load processor first as it might be needed for sample processing
        processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
        # Load model components
        deconstructed_model = DeconstructedFlorence2(MODEL_ID, device=DEVICE, dtype=COMPUTE_DTYPE, trust_remote_code=True)
        deconstructed_model.model.eval() # Set model to evaluation mode
        vision_encoder = deconstructed_model.vision_encoder # Access the vision encoder part
    except Exception as e:
        print(f"Error loading model or processor '{MODEL_ID}': {e}", file=sys.stderr)
        sys.exit(1)

    # 4. Determine Shapes and Dtypes (using one sample from original data)
    print("\nDetermining HDF5 shapes and dtypes...")
    try:
        # Use a sample from the *original* dataset to get shapes; subsetting doesn't change per-item shape
        sample_source = original_train_dataset if N_train_original > 0 else original_val_dataset
        if not sample_source:
             raise ValueError("Cannot determine shapes, both original splits are empty.")

        sample_data = next(iter(sample_source))
        sample_image = sample_data[IMAGE_COLUMN]

        # Handle potential non-PIL images in datasets library
        if not isinstance(sample_image, Image.Image):
             print(f"Warning: Sample image is not PIL Image, type is {type(sample_image)}. Attempting to load.")
             try:
                 if isinstance(sample_image, dict) and 'bytes' in sample_image and sample_image['bytes']:
                     from io import BytesIO
                     sample_image = Image.open(BytesIO(sample_image['bytes'])).convert("RGB")
                 elif isinstance(sample_image, str) and os.path.exists(sample_image): # If it's a path string
                      sample_image = Image.open(sample_image).convert("RGB")
                 # Add more loaders if necessary based on dataset format
                 else:
                     # Attempt default conversion if possible
                     sample_image = sample_image.convert("RGB") # datasets Image object might have convert()
             except Exception as load_err:
                  print(f"Failed to load or convert sample image: {load_err}", file=sys.stderr)
                  print("Ensure the image column contains processable image data (PIL Image, path, or bytes).")
                  sys.exit(1)

        # Get shapes/dtypes for encoded features and labels_to_store ONLY
        encoded_shape_tpl, encoded_storage_dtype, label_storage_dtypes = get_shapes_and_dtypes(
            processor, vision_encoder, sample_image, LABEL_COLUMNS_TO_STORE, DEVICE, COMPUTE_DTYPE
        )
        print(f"  Encoded shape (SeqLen, Channels): {encoded_shape_tpl}, storage dtype: {encoded_storage_dtype}")
        for label_col, dtype in label_storage_dtypes.items():
             print(f"  Label '{label_col}' storage dtype: {dtype}")

    except Exception as e:
        print(f"Error determining shapes/dtypes: {e}", file=sys.stderr)
        # Add more specific error handling if needed
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # 5. Initialize HDF5 File - Save indices
    print(f"\nInitializing HDF5 file (pre-allocated for subset): {HDF5_OUTPUT_PATH}")
    os.makedirs(os.path.dirname(HDF5_OUTPUT_PATH), exist_ok=True)
    try:
        with h5py.File(HDF5_OUTPUT_PATH, 'w') as file:
            # Add top-level attributes for context
            file.attrs['dataset_name'] = DATASET_NAME
            file.attrs['dataset_config'] = DATASET_CONFIG if DATASET_CONFIG else 'N/A'
            file.attrs['stratify_columns'] = str(STRATIFY_COLUMNS) # Store as string
            # Store both target sample counts
            file.attrs['target_samples_per_stratum_train'] = TARGET_SAMPLES_PER_STRATUM_TRAIN
            file.attrs['target_samples_per_stratum_val'] = TARGET_SAMPLES_PER_STRATUM_VAL
            file.attrs['model_id'] = MODEL_ID
            file.attrs['storage_float_dtype'] = str(FLOAT_STORAGE_DTYPE)
            file.attrs['storage_label_dtype'] = str(LABEL_STORAGE_DTYPE)
            file.attrs['random_seed'] = RANDOM_SEED # Good practice to store seed used


            train_group = file.create_group('training')
            val_group = file.create_group('validation')

            # Pass indices list along with other info
            for group, n_samples_subset, split_name, indices_list in [
                (train_group, N_train_subset, 'training', train_indices),
                (val_group, N_val_subset, 'validation', val_indices)
            ]:
                group.attrs['subset_size'] = n_samples_subset
                if n_samples_subset == 0:
                    print(f"  Skipping dataset creation for empty '{split_name}' subset.")
                    continue
                print(f"  Creating datasets for '{split_name}' group (subset size: {n_samples_subset})...")

                # --- Save Original Indices ---
                if indices_list: # Only save if list is not empty
                    indices_dset = group.create_dataset(
                        'original_indices',
                        data=np.array(indices_list, dtype=np.int64), # Use efficient int type
                        compression=hdf5plugin.Zstd(clevel=ZSTD_CLEVEL) # Compress indices too
                    )
                    indices_dset.attrs['description'] = f"Indices of these samples within the original '{split_name}' split of {DATASET_NAME}/{DATASET_CONFIG}."
                    print(f"    Saved 'original_indices' dataset with shape {indices_dset.shape}")
                else:
                     print(f"    Skipping 'original_indices' dataset for empty subset '{split_name}'.")


                # Encoded data
                encoded_chunks = (min(BATCH_SIZE, n_samples_subset),) + encoded_shape_tpl
                print(f"    Using encoded chunk shape: {encoded_chunks}")

                group.create_dataset(
                    'encoded',
                    shape=(n_samples_subset,) + encoded_shape_tpl,
                    dtype=encoded_storage_dtype,
                    chunks=encoded_chunks, # Use corrected chunks
                    compression=hdf5plugin.Zstd(clevel=ZSTD_CLEVEL)
                )


                # Labels group
                labels_group = group.create_group('labels')
                label_chunks = (min(BATCH_SIZE * 10, n_samples_subset),)
                for label_col, label_dtype in label_storage_dtypes.items():
                     labels_group.create_dataset(
                         label_col,
                         shape=(n_samples_subset,),
                         dtype=label_dtype,
                         chunks=label_chunks,
                         compression=hdf5plugin.Zstd(clevel=ZSTD_CLEVEL)
                     )


            # 6. Process and Store Subsets
            print("\nStarting subset data processing and storage...")

            if N_train_subset > 0:
                process_and_store_split(
                    train_subset, train_group, processor, vision_encoder,
                    IMAGE_COLUMN, LABEL_COLUMNS_TO_STORE, BATCH_SIZE, NUM_WORKERS, PIN_MEMORY,
                    DEVICE, COMPUTE_DTYPE, 'training'
                )
            else:
                print("Skipping processing for empty 'training' subset.")

            if N_val_subset > 0:
                process_and_store_split(
                    val_subset, val_group, processor, vision_encoder,
                    IMAGE_COLUMN, LABEL_COLUMNS_TO_STORE, BATCH_SIZE, NUM_WORKERS, PIN_MEMORY,
                    DEVICE, COMPUTE_DTYPE, 'validation'
                )
            else:
                 print("Skipping processing for empty 'validation' subset.")


    except Exception as e:
        print(f"\nError during HDF5 creation or data processing/storage: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        # Clean up potentially incomplete HDF5 file
        if os.path.exists(HDF5_OUTPUT_PATH):
            print(f"Removing potentially incomplete file: {HDF5_OUTPUT_PATH}", file=sys.stderr)
            # os.remove(HDF5_OUTPUT_PATH) # Be cautious with auto-removal
        sys.exit(1)

    # 7. Cleanup (Handled by 'with' statement)
    print(f"\nSubset dataset curation complete. Output saved to: {HDF5_OUTPUT_PATH}")
