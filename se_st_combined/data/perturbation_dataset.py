"""
Perturbation Dataset for SE+ST Training

Loads single-cell perturbation data from H5 files based on TOML configuration.
Handles control/perturbation pairing for perturbation prediction tasks.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set
import glob
import random

import torch
from torch.utils.data import Dataset
import h5py
import numpy as np
import tomli

logger = logging.getLogger(__name__)


class AnnDataH5Reader:
    """Helper class to read AnnData H5 files with categorical data support."""
    
    def __init__(self, h5_path: str):
        self.h5_path = Path(h5_path)
        self.h5_file = None
        
    def __enter__(self):
        self.h5_file = h5py.File(self.h5_path, 'r')
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.h5_file:
            self.h5_file.close()
    
    def read_categorical(self, group_path: str) -> np.ndarray:
        """Read categorical data from H5 group."""
        try:
            group = self.h5_file[group_path]
            if isinstance(group, h5py.Group):
                if 'codes' in group and 'categories' in group:
                    # Categorical format: codes index into categories
                    codes = np.array(group['codes'])
                    categories = np.array(group['categories'])
                    # Decode bytes to strings
                    if categories.dtype.kind == 'S':
                        categories = np.array([c.decode('utf-8') for c in categories])
                    # Map codes to categories
                    result = categories[codes]
                    return result
                elif 'values' in group:
                    # Direct values
                    values = np.array(group['values'])
                    if values.dtype.kind == 'S':
                        values = np.array([v.decode('utf-8') for v in values])
                    return values
            else:
                # Direct dataset
                data = np.array(group)
                if data.dtype.kind == 'S':
                    data = np.array([d.decode('utf-8') for d in data])
                return data
        except Exception as e:
            logger.error(f"Error reading categorical from {group_path}: {e}")
            raise
    
    def read_expression_matrix(self) -> np.ndarray:
        """Read expression matrix X."""
        if 'X' in self.h5_file:
            X = self.h5_file['X']
            # Check if it's a sparse matrix
            attrs = dict(X.attrs) if hasattr(X, 'attrs') else {}
            if attrs.get('encoding-type') == 'csr_matrix':
                # Handle CSR sparse matrix
                raise NotImplementedError("CSR sparse matrix not yet supported. Use obsm/X_hvg instead.")
            else:
                # Dense matrix
                X_array = np.array(X)
                if X_array.ndim != 2:
                    raise ValueError(f"Expected 2D matrix, got shape {X_array.shape}")
                return X_array
        else:
            raise KeyError("No expression matrix 'X' found in H5 file")
    
    def get_n_cells(self) -> int:
        """Get number of cells."""
        return self.h5_file['X'].shape[0]
    
    def get_n_genes(self) -> int:
        """Get number of genes."""
        return self.h5_file['X'].shape[1]


class PerturbationDataset(Dataset):
    """
    Dataset for loading single-cell perturbation data from H5 files.
    Creates control/perturbation pairs for training.
    
    Args:
        toml_config_path: Path to TOML configuration file
        perturbation_features_file: Path to perturbation embedding features (e.g., ESM2)
        split: Data split - 'train', 'val', or 'test'
        batch_col: Column name for batch variable
        pert_col: Column name for perturbation/gene target
        cell_type_key: Key for cell type annotation
        control_pert: Name of control perturbation (e.g., 'non-targeting')
        n_ctrl_samples: Number of control cells to sample per perturbation
        n_pert_samples: Number of perturbed cells to sample per pair
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
        n_ctrl_samples: int = 20,
        n_pert_samples: int = 20,
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
        self.n_ctrl_samples = n_ctrl_samples
        self.n_pert_samples = n_pert_samples
        
        # Determine data directory
        if data_dir is None:
            self.data_dir = self.toml_config_path.parent
        else:
            self.data_dir = Path(data_dir)
        
        # Load configuration
        self.config = self._load_toml_config()
        
        # Load perturbation embeddings
        self.pert_embeddings = self._load_perturbation_embeddings()
        
        # Load data from H5 files and create pairs
        self.pairs = self._load_and_create_pairs()
        
        logger.info(f"Created {len(self)} control/perturbation pairs for split '{split}'")
    
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
    
    def _load_and_create_pairs(self) -> List[Dict]:
        """Load data from H5 files and create control/perturbation pairs."""
        pairs = []
        
        # Get dataset configurations
        datasets = self.config.get("datasets", {})
        zeroshot_config = self.config.get("zeroshot", {})
        
        for dataset_name, file_pattern in datasets.items():
            logger.info(f"Processing dataset: {dataset_name}")
            
            # Expand file pattern to get actual files
            h5_files = self._expand_file_pattern(file_pattern)
            
            if not h5_files:
                logger.warning(f"No H5 files found for pattern: {file_pattern}")
                continue
            
            logger.info(f"Found {len(h5_files)} H5 files")
            
            # Load each H5 file
            for h5_file in h5_files:
                cell_type = h5_file.stem
                full_key = f"{dataset_name}.{cell_type}"
                
                # Check if this cell type should be in current split
                zeroshot_split = zeroshot_config.get(full_key, None)
                
                should_include = False
                if self.split == "train":
                    should_include = (zeroshot_split != "test")
                elif self.split == "val":
                    should_include = (zeroshot_split == "val")
                elif self.split == "test":
                    should_include = (zeroshot_split == "test")
                
                if not should_include:
                    logger.info(f"Skipping {cell_type} for split {self.split}")
                    continue
                
                # Load and create pairs from this file
                logger.info(f"Loading {h5_file.name} for {self.split} split")
                file_pairs = self._load_and_pair_single_h5(h5_file, cell_type)
                pairs.extend(file_pairs)
                logger.info(f"Created {len(file_pairs)} pairs from {h5_file.name}")
        
        return pairs
    
    def _load_and_pair_single_h5(self, h5_file: Path, cell_type: str) -> List[Dict]:
        """
        Load data from a single H5 file and create control/perturbation pairs.
        
        Returns:
            List of pairs, each containing control and perturbed cell sets
        """
        pairs = []
        
        try:
            with AnnDataH5Reader(h5_file) as reader:
                # Load expression matrix
                X = reader.read_expression_matrix()
                n_cells, n_genes = X.shape
                
                # Load metadata
                target_genes = reader.read_categorical(f'obs/{self.pert_col}')
                
                # Try to load batch info
                try:
                    batches = reader.read_categorical(f'obs/{self.batch_col}')
                except:
                    batches = np.array(['batch_0'] * n_cells)
                    logger.warning(f"No batch info found, using single batch")
                
                # Organize cells by perturbation and batch
                cell_indices_by_pert_batch = {}
                
                for i in range(n_cells):
                    pert = target_genes[i]
                    batch = batches[i]
                    key = (pert, batch)
                    
                    if key not in cell_indices_by_pert_batch:
                        cell_indices_by_pert_batch[key] = []
                    cell_indices_by_pert_batch[key].append(i)
                
                # Get all unique perturbations (excluding control)
                all_perts = set(target_genes)
                perturbations = [p for p in all_perts if p != self.control_pert]
                
                logger.info(f"Found {len(perturbations)} perturbations and {len(all_perts)} total conditions")
                
                # Create pairs for each perturbation
                for pert in perturbations:
                    # Decode bytes to string if needed
                    pert_str = pert.decode('utf-8') if isinstance(pert, bytes) else str(pert)
                    
                    # Skip if no embedding available
                    if pert_str not in self.pert_embeddings:
                        # Try some common variations
                        if pert_str.upper() not in self.pert_embeddings and pert_str.lower() not in self.pert_embeddings:
                            logger.warning(f"No embedding found for {pert_str}, skipping")
                            continue
                        elif pert_str.upper() in self.pert_embeddings:
                            pert_str = pert_str.upper()
                        else:
                            pert_str = pert_str.lower()
                    
                    # Find all batches that have this perturbation
                    pert_batches = [batch for (p, batch) in cell_indices_by_pert_batch.keys() if p == pert]
                    
                    for batch in pert_batches:
                        # Get perturbed cells
                        pert_key = (pert, batch)
                        if pert_key not in cell_indices_by_pert_batch:
                            continue
                        pert_indices = cell_indices_by_pert_batch[pert_key]
                        
                        # Get control cells from same batch
                        ctrl_key = (self.control_pert, batch)
                        if ctrl_key not in cell_indices_by_pert_batch:
                            # Try to find control from any batch
                            ctrl_indices = []
                            for (p, b) in cell_indices_by_pert_batch.keys():
                                if p == self.control_pert:
                                    ctrl_indices.extend(cell_indices_by_pert_batch[(p, b)])
                            if not ctrl_indices:
                                logger.warning(f"No control cells found for {pert} in {batch}")
                                continue
                        else:
                            ctrl_indices = cell_indices_by_pert_batch[ctrl_key]
                        
                        # Sample cells
                        n_ctrl_available = len(ctrl_indices)
                        n_pert_available = len(pert_indices)
                        
                        n_ctrl = min(self.n_ctrl_samples, n_ctrl_available)
                        n_pert = min(self.n_pert_samples, n_pert_available)
                        
                        if n_ctrl < 5 or n_pert < 5:
                            # Skip if too few cells
                            continue
                        
                        # Sample with replacement if needed
                        sampled_ctrl_indices = np.random.choice(ctrl_indices, size=n_ctrl, replace=(n_ctrl > n_ctrl_available))
                        sampled_pert_indices = np.random.choice(pert_indices, size=n_pert, replace=(n_pert > n_pert_available))
                        
                        # Create pair
                        pair = {
                            'ctrl_cell_emb': torch.tensor(X[sampled_ctrl_indices], dtype=torch.float32),
                            'pert_cell_emb': torch.tensor(X[sampled_pert_indices], dtype=torch.float32),
                            'pert_embedding': self.pert_embeddings[pert_str],
                            'perturbation': pert_str,
                            'cell_type': cell_type,
                            'batch': batch,
                        }
                        
                        pairs.append(pair)
        
        except Exception as e:
            logger.error(f"Error loading {h5_file}: {e}")
            logger.exception("Full traceback:")
        
        return pairs
    
    def __len__(self) -> int:
        return len(self.pairs)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single control/perturbation pair."""
        pair = self.pairs[idx]
        
        # Pairs are already in the correct format with tensors
        return pair


def collate_perturbation_batch(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """
    Collate function for perturbation dataset.
    Handles variable-sized cell sets by concatenating them.
    
    Args:
        batch: List of pairs from dataset
        
    Returns:
        Dictionary with batched tensors
    """
    if not batch:
        return {}
    
    # Concatenate control cells from all pairs in batch
    ctrl_cells = torch.cat([pair['ctrl_cell_emb'] for pair in batch], dim=0)
    
    # Concatenate perturbed cells from all pairs in batch
    pert_cells = torch.cat([pair['pert_cell_emb'] for pair in batch], dim=0)
    
    # Stack perturbation embeddings
    pert_embeddings = torch.stack([pair['pert_embedding'] for pair in batch])
    
    # Collect metadata
    perturbations = [pair['perturbation'] for pair in batch]
    cell_types = [pair['cell_type'] for pair in batch]
    batches = [pair['batch'] for pair in batch]
    
    output = {
        'ctrl_cell_emb': ctrl_cells,
        'pert_cell_emb': pert_cells,
        'pert_embedding': pert_embeddings,
        'perturbation': perturbations,
        'cell_type': cell_types,
        'batch': batches,
    }
    
    return output

