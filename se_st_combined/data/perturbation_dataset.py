"""
Perturbation Dataset for SE+ST Training

Loads single-cell perturbation data from H5 files based on TOML configuration.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import glob

import torch
from torch.utils.data import Dataset
import h5py
import numpy as np
import tomli

logger = logging.getLogger(__name__)


class PerturbationDataset(Dataset):
    """
    Dataset for loading single-cell perturbation data from H5 files.
    
    Args:
        toml_config_path: Path to TOML configuration file
        perturbation_features_file: Path to perturbation embedding features (e.g., ESM2)
        split: Data split - 'train', 'val', or 'test'
        batch_col: Column name for batch variable
        pert_col: Column name for perturbation/gene target
        cell_type_key: Key for cell type annotation
        control_pert: Name of control perturbation (e.g., 'non-targeting')
    """
    
    def __init__(
        self,
        toml_config_path: str,
        perturbation_features_file: str,
        split: str = "train",
        batch_col: str = "batch_var",
        pert_col: str = "target_gene",
        cell_type_key: str = "cell_type",
        control_pert: str = "non-targeting",
        data_dir: Optional[str] = None,
    ):
        super().__init__()
        self.toml_config_path = Path(toml_config_path)
        self.perturbation_features_file = Path(perturbation_features_file)
        self.split = split
        self.batch_col = batch_col
        self.pert_col = pert_col
        self.cell_type_key = cell_type_key
        self.control_pert = control_pert
        
        # Determine data directory
        if data_dir is None:
            self.data_dir = self.toml_config_path.parent
        else:
            self.data_dir = Path(data_dir)
        
        # Load configuration
        self.config = self._load_toml_config()
        
        # Load perturbation embeddings
        self.pert_embeddings = self._load_perturbation_embeddings()
        
        # Load data from H5 files
        self.data = self._load_h5_data()
        
        logger.info(f"Loaded {len(self)} samples for split '{split}'")
    
    def _load_toml_config(self) -> Dict:
        """Load TOML configuration file."""
        logger.info(f"Loading TOML config from {self.toml_config_path}")
        
        with open(self.toml_config_path, "rb") as f:
            config = tomli.load(f)
        
        logger.info(f"Config datasets: {list(config.get('datasets', {}).keys())}")
        return config
    
    def _load_perturbation_embeddings(self) -> Dict[str, torch.Tensor]:
        """Load perturbation embeddings (e.g., ESM2 protein embeddings)."""
        logger.info(f"Loading perturbation embeddings from {self.perturbation_features_file}")
        
        if not self.perturbation_features_file.exists():
            logger.warning(f"Perturbation features file not found: {self.perturbation_features_file}")
            return {}
        
        pert_embeddings = torch.load(self.perturbation_features_file)
        logger.info(f"Loaded embeddings for {len(pert_embeddings)} perturbations")
        
        return pert_embeddings
    
    def _expand_file_pattern(self, pattern: str) -> List[Path]:
        """
        Expand file pattern with glob syntax.
        
        Example: "{competition_train,k562}.h5" -> ["competition_train.h5", "k562.h5"]
        """
        # Replace pattern with actual directory
        pattern = pattern.replace("/content/state/competition_support_set/", str(self.data_dir) + "/")
        
        # Handle brace expansion manually if needed
        if "{" in pattern and "}" in pattern:
            # Extract the pattern inside braces
            import re
            match = re.search(r'\{([^}]+)\}', pattern)
            if match:
                options = match.group(1).split(',')
                base_pattern = pattern[:match.start()] + "{}" + pattern[match.end():]
                expanded = [Path(base_pattern.format(opt.strip())) for opt in options]
                return [p for p in expanded if p.exists()]
        
        # Use glob for standard patterns
        expanded = list(Path(self.data_dir).glob(pattern))
        return expanded
    
    def _load_h5_data(self) -> List[Dict]:
        """Load data from H5 files based on TOML configuration."""
        data_samples = []
        
        # Get dataset configurations
        datasets = self.config.get("datasets", {})
        training_config = self.config.get("training", {})
        zeroshot_config = self.config.get("zeroshot", {})
        
        for dataset_name, file_pattern in datasets.items():
            logger.info(f"Processing dataset: {dataset_name}")
            logger.info(f"File pattern: {file_pattern}")
            
            # Expand file pattern to get actual files
            h5_files = self._expand_file_pattern(file_pattern)
            
            if not h5_files:
                logger.warning(f"No H5 files found for pattern: {file_pattern}")
                continue
            
            logger.info(f"Found {len(h5_files)} H5 files: {[f.name for f in h5_files]}")
            
            # Load each H5 file
            for h5_file in h5_files:
                cell_type = h5_file.stem  # e.g., "k562" from "k562.h5"
                full_key = f"{dataset_name}.{cell_type}"
                
                # Check if this cell type should be in current split
                zeroshot_split = zeroshot_config.get(full_key, None)
                
                # Determine if this file belongs to current split
                should_include = False
                if self.split == "train":
                    # Include if not in zeroshot test
                    should_include = (zeroshot_split != "test")
                elif self.split == "val":
                    should_include = (zeroshot_split == "val")
                elif self.split == "test":
                    should_include = (zeroshot_split == "test")
                
                if not should_include:
                    logger.info(f"Skipping {cell_type} for split {self.split}")
                    continue
                
                # Load H5 file
                logger.info(f"Loading {h5_file.name} for {self.split} split")
                file_samples = self._load_single_h5(h5_file, cell_type)
                data_samples.extend(file_samples)
                logger.info(f"Loaded {len(file_samples)} samples from {h5_file.name}")
        
        return data_samples
    
    def _load_single_h5(self, h5_file: Path, cell_type: str) -> List[Dict]:
        """Load data from a single H5 file."""
        samples = []
        
        try:
            with h5py.File(h5_file, "r") as f:
                # Try to load data in standard formats
                # Common keys: 'X', 'obs', 'var', 'obsm', 'varm', 'uns'
                
                # Load expression matrix
                if "X" in f:
                    X = np.array(f["X"])
                elif "data" in f:
                    X = np.array(f["data"])
                else:
                    logger.warning(f"No expression data found in {h5_file.name}")
                    return samples
                
                # Load metadata
                obs_data = {}
                if "obs" in f:
                    obs_group = f["obs"]
                    for key in obs_group.keys():
                        obs_data[key] = np.array(obs_group[key])
                
                # Create samples (one per cell)
                n_cells = X.shape[0]
                for i in range(n_cells):
                    sample = {
                        "expression": torch.tensor(X[i], dtype=torch.float32),
                        "cell_type": cell_type,
                    }
                    
                    # Add perturbation info if available
                    if self.pert_col in obs_data:
                        pert_name = obs_data[self.pert_col][i]
                        if isinstance(pert_name, bytes):
                            pert_name = pert_name.decode("utf-8")
                        sample["perturbation"] = pert_name
                        
                        # Add perturbation embedding
                        if pert_name in self.pert_embeddings:
                            sample["pert_embedding"] = self.pert_embeddings[pert_name]
                    
                    # Add batch info if available
                    if self.batch_col in obs_data:
                        sample["batch"] = obs_data[self.batch_col][i]
                    
                    samples.append(sample)
        
        except Exception as e:
            logger.error(f"Error loading {h5_file}: {e}")
            logger.exception("Full traceback:")
        
        return samples
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single sample."""
        sample = self.data[idx]
        
        # Convert numpy arrays to tensors if needed
        output = {}
        for key, value in sample.items():
            if isinstance(value, np.ndarray):
                output[key] = torch.tensor(value, dtype=torch.float32)
            elif isinstance(value, torch.Tensor):
                output[key] = value
            else:
                output[key] = value
        
        return output


def collate_perturbation_batch(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """
    Collate function for perturbation dataset.
    
    Args:
        batch: List of samples from dataset
        
    Returns:
        Dictionary with batched tensors
    """
    if not batch:
        return {}
    
    # Initialize output dictionary
    output = {}
    
    # Get all keys from first sample
    keys = batch[0].keys()
    
    for key in keys:
        values = [sample[key] for sample in batch]
        
        # Stack tensors
        if isinstance(values[0], torch.Tensor):
            try:
                output[key] = torch.stack(values)
            except:
                # If stacking fails, keep as list
                output[key] = values
        else:
            # Keep non-tensor values as list
            output[key] = values
    
    return output

