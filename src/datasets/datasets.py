import h5py
import hdf5plugin
import torch
from torch.utils.data import Dataset, DataLoader, get_worker_info
import logging
from typing import List, Optional, Dict, Union, Tuple
import time
import os

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
log = logging.getLogger(__name__)

def dataset_worker_init_fn(worker_id: int) -> None:
    """
    Worker initialization function for DataLoader.
    Opens the HDF5 file once per worker to avoid repeated file opening.

    Args:
        worker_id: ID of the worker process.
    """
    worker_info = get_worker_info()
    if worker_info is not None:
        dataset = worker_info.dataset  # Get the dataset copy in this worker process
        try:
            # Open the file and store the handle in the dataset instance
            dataset.file_handle = h5py.File(dataset.hdf5_path, 'r')
            log.debug(f"Worker {worker_id}: Successfully opened file {dataset.hdf5_path}")
        except Exception as e:
            log.error(f"Worker {worker_id}: Failed to open HDF5 file {dataset.hdf5_path}: {e}")
            # Set handle to None so __getitem__ knows init failed for this worker
            dataset.file_handle = None


class FairFaceDataset(Dataset):
    """
    A PyTorch Dataset for loading FairFace features and labels from an HDF5 file.

    Provides flexible label handling and efficient file I/O with DataLoader workers.

    Args:
        hdf5_path: Path to the HDF5 file.
        mode: The group key within the HDF5 file (e.g., 'training', 'validation').
        return_labels: Which labels to return. Can be:
                     - None or []: Return features only.
                     - ['all']: Return all available labels under the 'labels' group.
                     - List of specific keys (e.g., ['age', 'race']): Return only those labels.
                       Defaults to ['all'].
    """
    def __init__(self,
                 hdf5_path: str,
                 mode: str = 'training',
                 return_labels: Optional[List[str]] = ['all']): # Default to returning all labels

        self.hdf5_path = hdf5_path
        self.mode = mode
        self._requested_labels = return_labels if return_labels is not None else []
        self.file_handle = None  # Will be set by worker_init_fn or in __getitem__
        self.label_keys_to_load = [] # Final list of keys to actually load

        log.info(f"Initializing FairFaceDataset: path={hdf5_path}, mode={mode}, requested_labels={self._requested_labels}")

        # Open file briefly to get length and validate/discover label keys
        try:
            with h5py.File(self.hdf5_path, 'r') as f:
                log.debug(f"Checking HDF5 structure for mode '{self.mode}'...")
                if self.mode not in f:
                    raise ValueError(f"Mode '{self.mode}' not found in HDF5 file groups: {list(f.keys())}")

                group = f[self.mode]

                # Check for encoded features dataset
                if 'encoded' not in group:
                    raise ValueError(f"'encoded' dataset not found in HDF5 group '{self.mode}'")

                # Get dataset length
                self.length = group['encoded'].shape[0]
                log.info(f"Dataset contains {self.length} items in mode '{self.mode}'")

                # Determine which labels to load based on requested_labels
                if not self._requested_labels: # If None or []
                    log.info("Labels explicitly disabled. Will return features only.")
                    self.label_keys_to_load = []
                else:
                    # Check if the 'labels' group exists
                    if 'labels' not in group:
                        log.warning(f"'labels' group not found in HDF5 group '{self.mode}'. Cannot return labels.")
                        self.label_keys_to_load = []
                    else:
                        labels_group = group['labels']
                        available_label_keys = list(labels_group.keys())
                        log.debug(f"Available labels in group '{self.mode}': {available_label_keys}")

                        if self._requested_labels == ['all']:
                            self.label_keys_to_load = available_label_keys
                            log.info(f"Configured to load all available labels: {self.label_keys_to_load}")
                        else:
                            # Validate the specifically requested keys
                            invalid_keys = [key for key in self._requested_labels if key not in available_label_keys]
                            if invalid_keys:
                                raise ValueError(f"Requested label_keys {invalid_keys} not found in available keys {available_label_keys} for mode '{self.mode}'.")
                            self.label_keys_to_load = self._requested_labels
                            log.info(f"Configured to load specified labels: {self.label_keys_to_load}")

        except Exception as e:
            log.exception(f"Failed during dataset initialization from {self.hdf5_path} mode '{self.mode}'")
            raise # Re-raise the exception after logging

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        """
        Get an item from the dataset.

        Args:
            idx: Index of the item to get.

        Returns:
            If no labels were requested: features tensor.
            If labels were requested: Tuple of (features tensor, labels dictionary).
        """
        # Check if the file handle exists (set by worker_init_fn or opened here)
        worker_info = get_worker_info()
        is_worker = worker_info is not None
        pid = os.getpid()
        log_prefix = f"Worker {worker_info.id} (PID {pid})" if is_worker else f"Main Process (PID {pid})"

        need_to_close = False
        current_file_handle = None

        try:
            if hasattr(self, 'file_handle') and self.file_handle is not None:
                # Use handle created by worker_init_fn
                current_file_handle = self.file_handle
                # log.debug(f"{log_prefix}: Using existing file handle for item {idx}.")
            else:
                # Handle case for num_workers=0 or if worker_init_fn failed
                if is_worker:
                    # Worker init must have failed for this worker
                    log.error(f"{log_prefix}: File handle is missing for item {idx}. Worker init likely failed.")
                    # Depending on desired robustness, could return dummy data or raise error
                    raise IOError(f"HDF5 file handle missing in worker {worker_info.id}. Cannot read item {idx}.")
                else:
                    # Main process (num_workers=0) - open file temporarily
                    log.debug(f"{log_prefix}: Opening file on demand for item {idx} (num_workers=0?).")
                    current_file_handle = h5py.File(self.hdf5_path, 'r')
                    need_to_close = True # Mark for closing in finally block

            if current_file_handle is None: # Double check if opening failed
                raise IOError(f"Failed to obtain valid HDF5 file handle in {log_prefix}.")

            # Access the correct group within the HDF5 file
            group = current_file_handle[self.mode]

            # --- Always load features ---
            # log.debug(f"{log_prefix}: Reading features for item {idx}.")
            # Use read_direct for potentially better performance with some storage layouts
            # features_np = np.zeros(self._encoded_shape, dtype=self._encoded_dtype) # Requires shape/dtype stored in __init__
            # group['encoded'].read_direct(features_np, source_sel=np.s_[idx])
            # features = torch.from_numpy(features_np).float()
            # Simple slicing is usually fine and easier:
            features = torch.tensor(group['encoded'][idx], dtype=torch.float32)
            # log.debug(f"{log_prefix}: Features read successfully for item {idx}.")


            # --- Load labels if requested ---
            if not self.label_keys_to_load:
                # log.debug(f"{log_prefix}: Returning features only for item {idx}.")
                return features
            else:
                # log.debug(f"{log_prefix}: Reading labels for item {idx}: {self.label_keys_to_load}")
                labels_dict = {}
                labels_group = group['labels']
                for key in self.label_keys_to_load:
                    # log.debug(f"{log_prefix}: Reading label '{key}' for item {idx}.")
                    # Use simple slicing:
                    labels_dict[key] = torch.tensor(labels_group[key][idx], dtype=torch.long) # Assuming labels are integer IDs

                # log.debug(f"{log_prefix}: Labels read successfully for item {idx}.")
                return features, labels_dict

        except Exception as e:
            log.exception(f"{log_prefix}: CRITICAL - Error reading item {idx} from {self.hdf5_path} mode '{self.mode}'")
            # Re-raise the exception so DataLoader knows the worker failed
            raise

        finally:
            # Close the file handle only if it was opened within this __getitem__ call
            if need_to_close and current_file_handle is not None:
                log.debug(f"{log_prefix}: Closing file handle opened on demand for item {idx}.")
                try:
                    current_file_handle.close()
                except Exception as e_close:
                    log.warning(f"{log_prefix}: Exception closing temporary file handle: {e_close}")


    def __del__(self) -> None:
        """Clean up the file handle if this object is deleted (e.g., end of worker process)."""
        # This ensures the handle opened by worker_init_fn gets closed when the worker exits
        pid = os.getpid()
        if hasattr(self, 'file_handle') and self.file_handle is not None:
            log.debug(f"Process {pid}: __del__ called for Dataset object. Closing HDF5 file handle for {self.hdf5_path}.")
            try:
                self.file_handle.close()
            except Exception as e: # HDF5 file might already be closed etc.
                log.warning(f"Process {pid}: Exception closing HDF5 file in __del__: {e}")
            self.file_handle = None


# --- Testing ---
if __name__ == "__main__":
    # Set log level for detailed testing output
    log.setLevel(logging.DEBUG)
    logging.getLogger('h5py').setLevel(logging.WARNING) # Keep h5py logs quieter unless needed

    # IMPORTANT: Update this path to your actual HDF5 file location
    HDF5_PATH = '/home/ozanbayiz/irve/output_data/fairface_latent_stratified.hdf5'

    if not os.path.exists(HDF5_PATH):
        log.error(f"Test HDF5 file not found at: {HDF5_PATH}")
        print(f"Please update the HDF5_PATH variable in the script.")
        exit(1)

    print(f"\n{'='*80}")
    print("FAIRFACE DATASET IMPLEMENTATION TESTING")
    print(f"{'='*80}")

    def run_functional_test(name, dataset_params, loader_params, expected_labels=None):
        """Test functionality: create dataset/loader, get one batch, check output."""
        print(f"\n--- Functional Test: {name} ---")
        print(f"Dataset params: {dataset_params}")
        print(f"Loader params: {loader_params}")

        # Add worker_init_fn if using workers
        if loader_params.get('num_workers', 0) > 0:
            loader_params['worker_init_fn'] = dataset_worker_init_fn
            # persistent_workers is recommended with num_workers > 0
            if 'persistent_workers' not in loader_params:
                 loader_params['persistent_workers'] = True

        try:
            dataset = FairFaceDataset(HDF5_PATH, **dataset_params)
            loader = DataLoader(dataset, **loader_params)

            print(f"Total items in dataset: {len(dataset)}")

            # Get one batch
            start_time = time.time()
            batch = next(iter(loader))
            elapsed = time.time() - start_time
            print(f"First batch loaded in {elapsed:.4f} seconds")

            # Check output structure and labels
            if expected_labels is None or expected_labels == []:
                # Expecting only features tensor
                assert isinstance(batch, torch.Tensor), f"Expected Tensor, got {type(batch)}"
                print(f"Returned: features tensor of shape {batch.shape}, dtype {batch.dtype}")
            else:
                # Expecting (features, labels_dict) tuple
                assert isinstance(batch, (list, tuple)) and len(batch) == 2, f"Expected Tuple(Tensor, Dict), got {type(batch)}"
                features, labels_dict = batch
                assert isinstance(features, torch.Tensor), f"Expected features Tensor, got {type(features)}"
                assert isinstance(labels_dict, dict), f"Expected labels Dict, got {type(labels_dict)}"
                print(f"Returned: features tensor shape {features.shape}, dtype {features.dtype}")
                print(f"Returned: labels dict with keys: {list(labels_dict.keys())}")

                # Validate the keys in the returned dict
                returned_keys = set(labels_dict.keys())
                if expected_labels == ['all']:
                     # Can't know 'all' without reading file again, just check it's non-empty if labels group exists
                     if dataset.label_keys_to_load: # Check if __init__ determined labels should be loaded
                         assert returned_keys, "Expected label keys for ['all'], but got empty dict"
                         print(f"  (Labels dictionary contains expected keys: {list(returned_keys)})")
                     else:
                         assert not returned_keys, "Expected empty labels dict when labels group missing/empty"
                         print("  (Labels dictionary is correctly empty)")

                else:
                    expected_keys_set = set(expected_labels)
                    assert returned_keys == expected_keys_set, f"Expected label keys {expected_keys_set}, got {returned_keys}"
                    print(f"  (Labels dictionary contains expected keys: {list(returned_keys)})")

                # Check shapes/types of label tensors
                for key, val in labels_dict.items():
                    assert isinstance(val, torch.Tensor), f"Label '{key}' is not a Tensor, got {type(val)}"
                    assert val.shape[0] == loader_params['batch_size'], f"Label '{key}' batch size mismatch: {val.shape[0]} vs {loader_params['batch_size']}"
                    assert val.dtype == torch.long, f"Label '{key}' dtype is not torch.long, got {val.dtype}"
                    print(f"    - {key}: shape {val.shape}, dtype {val.dtype}")


            print(f"Functional Test '{name}' PASSED!")
            return True

        except Exception as e:
            print(f"!!! Functional Test '{name}' FAILED: {e}")
            import traceback
            traceback.print_exc()
            return False

    # Test Case 1: Features only (return_labels=None)
    run_functional_test(
        name="Features Only (return_labels=None)",
        dataset_params={'mode': 'training', 'return_labels': None},
        loader_params={'batch_size': 4, 'num_workers': 0},
        expected_labels=None
    )

    # Test Case 2: Features only (return_labels=[])
    run_functional_test(
        name="Features Only (return_labels=[])",
        dataset_params={'mode': 'training', 'return_labels': []},
        loader_params={'batch_size': 4, 'num_workers': 0},
        expected_labels=[]
    )

    # Test Case 3: All available labels (default)
    run_functional_test(
        name="All Labels (Default)",
        dataset_params={'mode': 'training', 'return_labels': ['all']}, # Explicitly pass ['all']
        loader_params={'batch_size': 4, 'num_workers': 0},
        expected_labels=['all'] # Special marker for check
    )

    # Test Case 4: Specific labels
    run_functional_test(
        name="Specific Labels ['age', 'gender']",
        dataset_params={'mode': 'training', 'return_labels': ['age', 'gender']},
        loader_params={'batch_size': 4, 'num_workers': 0},
        expected_labels=['age', 'gender']
    )

    # Test Case 5: All available labels with multiple workers
    run_functional_test(
        name="All Labels (2 Workers)",
        dataset_params={'mode': 'training', 'return_labels': ['all']},
        loader_params={'batch_size': 8, 'num_workers': 2}, # No shuffle for easier debugging if needed
        expected_labels=['all']
    )

    # Test Case 6: Specific labels with multiple workers
    run_functional_test(
        name="Specific Labels ['race'] (4 Workers)",
        dataset_params={'mode': 'training', 'return_labels': ['race']},
        loader_params={'batch_size': 8, 'num_workers': 4},
        expected_labels=['race']
    )

    # Test Case 7: Requesting a non-existent label (should fail in __init__)
    print("\n--- Expecting Initialization Error Below ---")
    try:
        run_functional_test(
            name="Non-existent Label Key",
            dataset_params={'mode': 'training', 'return_labels': ['age', 'non_existent']},
            loader_params={'batch_size': 4, 'num_workers': 0},
            expected_labels=[] # Won't get here
        )
    except ValueError as e:
        print(f"Caught expected ValueError during init: {e}")
        print("Test PASSED (Correctly failed initialization).")
    except Exception as e:
        print(f"!!! Test 'Non-existent Label Key' FAILED with unexpected error: {e}")

    print(f"\n{'='*80}")
    print("Testing finished.")
    print(f"{'='*80}")