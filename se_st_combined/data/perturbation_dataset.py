"""
SE+ST Combined Model - Perturbation Dataset

This module implements the data loading and pairing logic for the SE+ST combined model.
Heavily inspired by STATE's cell-load implementation.
"""

import logging
import glob
import re
from pathlib import Path
from typing import Dict, List, Set, Optional

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
import tomli


logger = logging.getLogger(__name__)


def safe_decode_array(arr) -> np.ndarray:
    """
    Decode any byte-strings in arr to UTF-8 and cast all entries to Python str.
    
    Args:
        arr: array-like of bytes or other objects
    Returns:
        np.ndarray[str]: decoded strings
    """
    decoded = []
    for x in arr:
        if isinstance(x, (bytes, bytearray)):
            decoded.append(x.decode("utf-8", errors="ignore"))
        else:
            decoded.append(str(x))
    return np.array(decoded, dtype=str)


class H5MetadataCache:
    """Cache for H5 file metadata to avoid repeated disk reads."""
    
    def __init__(
        self,
        h5_path: str,
        pert_col: str = "target_gene",
        cell_type_key: str = "cell_type",
        control_pert: str = "non-targeting",
        batch_col: str = "batch_var",
    ):
        self.h5_path = h5_path
        with h5py.File(h5_path, "r") as f:
            obs = f["obs"]
            
            # Read categories and decode to strings
            self.pert_categories = safe_decode_array(obs[pert_col]["categories"][:])
            self.cell_type_categories = safe_decode_array(obs[cell_type_key]["categories"][:])
            
            # Handle batch column (can be categorical or numeric)
            batch_ds = obs[batch_col]
            if "categories" in batch_ds:
                self.batch_categories = safe_decode_array(batch_ds["categories"][:])
                self.batch_codes = batch_ds["codes"][:].astype(np.int32)
            else:
                raw = batch_ds[:]
                self.batch_categories = raw.astype(str)
                self.batch_codes = raw.astype(np.int32)
            
            # Read codes
            self.pert_codes = obs[pert_col]["codes"][:].astype(np.int32)
            self.cell_type_codes = obs[cell_type_key]["codes"][:].astype(np.int32)
            
            # Find control perturbation code
            idx = np.where(self.pert_categories == control_pert)[0]
            if idx.size == 0:
                logger.warning(
                    f"control_pert='{control_pert}' not found in {pert_col} categories. "
                    f"Available: {sorted(set(self.pert_categories))[:10]}"
                )
                # Try case-insensitive match
                control_pert_lower = control_pert.lower()
                for i, cat in enumerate(self.pert_categories):
                    if cat.lower() == control_pert_lower:
                        idx = np.array([i])
                        logger.info(f"Found control using case-insensitive match: '{cat}'")
                        break
                
                if idx.size == 0:
                    raise ValueError(
                        f"control_pert='{control_pert}' not found (even case-insensitive). "
                        f"Available: {sorted(set(self.pert_categories))[:20]}"
                    )
            
            self.control_pert_code = int(idx[0])
            self.control_mask = self.pert_codes == self.control_pert_code
            
            self.n_cells = len(self.pert_codes)


class PerturbationDataset(Dataset):
    """
    Dataset class for loading perturbation data from H5 files.
    Each sample is a (control_cell, perturbed_cell, perturbation_embedding) pair.
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
        n_ctrl_samples: int = 5,
        n_pert_samples: int = 5,
        random_seed: int = 42,
        embed_key: str = None,  # Not used for now, but kept for compatibility
    ):
        """
        Initialize a perturbation dataset.
        
        Args:
            toml_config_path: Path to TOML configuration file
            perturbation_features_file: Path to perturbation features (ESM2 embeddings)
            split: Data split - 'train', 'val', or 'test'
            batch_col: H5 obs column for batches
            pert_col: H5 obs column for perturbations
            cell_type_key: H5 obs column for cell types
            control_pert: Perturbation treated as control
            n_ctrl_samples: Number of control cells to sample per perturbation
            n_pert_samples: Number of perturbed cells to sample per pair
            random_seed: Random seed for reproducibility
            embed_key: Key under obsm for embeddings (not used yet)
        """
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
        self.embed_key = embed_key
        self.rng = np.random.default_rng(random_seed)
        
        # Load configuration
        self.config = self._load_toml_config()
        
        # Load perturbation embeddings (ESM2 features)
        self.pert_embedding_map = self._load_perturbation_embeddings()
        
        # Storage for all pairs
        self.pairs: List[Dict] = []
        
        # Create pairs from H5 files
        self._setup_datasets()
        
        logger.info(f"✅ Dataset created with {len(self.pairs)} pairs for split '{self.split}'")
        if len(self.pairs) == 0:
            logger.error("❌ No pairs created!")
    
    def _load_toml_config(self) -> Dict:
        """Load TOML configuration file."""
        with open(self.toml_config_path, "rb") as f:
            config = tomli.load(f)
        return config
    
    def _load_perturbation_embeddings(self) -> Dict[str, torch.Tensor]:
        """
        Load perturbation embeddings from torch file.
        Returns a dictionary mapping perturbation names to embeddings.
        """
        pert_dict = torch.load(self.perturbation_features_file, map_location="cpu")
        logger.info(f"Loaded {len(pert_dict)} perturbation embeddings")
        logger.info(f"Sample keys: {list(pert_dict.keys())[:5]}")
        
        # Validate missing perturbations will be handled later
        return pert_dict
    
    def _expand_file_pattern(self, pattern: str) -> List[str]:
        """
        Expand brace patterns like {a,b,c} into multiple patterns.
        """
        def expand_single_brace(text: str) -> List[str]:
            match = re.search(r"\{([^}]+)\}", text)
            if not match:
                return [text]
            
            before = text[:match.start()]
            after = text[match.end():]
            options = match.group(1).split(",")
            
            results = []
            for option in options:
                new_text = before + option.strip() + after
                results.extend(expand_single_brace(new_text))
            return results
        
        return expand_single_brace(pattern)
    
    def _find_dataset_files(self, dataset_path: str) -> List[Path]:
        """Find all H5 files matching the dataset path pattern."""
        expanded_patterns = self._expand_file_pattern(dataset_path)
        files = []
        
        for pattern in expanded_patterns:
            if any(char in pattern for char in "*?[]{}"):
                # Use glob for patterns
                matching_files = sorted(glob.glob(pattern))
                files.extend([Path(f) for f in matching_files])
            else:
                # Direct path
                p = Path(pattern)
                if p.exists():
                    if p.is_file():
                        files.append(p)
                    else:
                        # Directory - search for H5 files
                        files.extend(sorted(p.glob("*.h5")))
                        files.extend(sorted(p.glob("*.h5ad")))
        
        return files
    
    def _setup_datasets(self):
        """
        Set up datasets according to TOML configuration and split.
        """
        for dataset_name, dataset_path in self.config.get("datasets", {}).items():
            # Get zeroshot configuration
            zeroshot_config = self.config.get("zeroshot", {})
            
            # Check if this dataset/celltype is designated for val or test
            zeroshot_key = f"{dataset_name}.*"
            is_zeroshot = False
            zeroshot_split = None
            
            # Check for specific celltype zeroshot
            for key, split_name in zeroshot_config.items():
                if key.startswith(dataset_name):
                    is_zeroshot = True
                    zeroshot_split = split_name
                    break
            
            # Determine if we should load this dataset for current split
            should_load = False
            if is_zeroshot:
                # Zeroshot: load if split matches zeroshot split
                should_load = (self.split == zeroshot_split)
            else:
                # Normal training: load for train split
                should_load = (self.split == "train")
            
            if not should_load:
                continue
            
            logger.info(f"Loading dataset: {dataset_name} for split: {self.split}")
            
            # Find H5 files
            files = self._find_dataset_files(dataset_path)
            logger.info(f"Found {len(files)} H5 files for {dataset_name}")
            
            for h5_file in files:
                logger.info(f"Processing {h5_file.name}...")
                try:
                    self._load_and_pair_single_h5(str(h5_file))
                except Exception as e:
                    logger.error(f"Error processing {h5_file}: {e}")
                    import traceback
                    traceback.print_exc()
    
    def _load_and_pair_single_h5(self, h5_path: str):
        """
        Load a single H5 file and create control/perturbation pairs.
        """
        # Load metadata
        cache = H5MetadataCache(
            h5_path,
            pert_col=self.pert_col,
            cell_type_key=self.cell_type_key,
            control_pert=self.control_pert,
            batch_col=self.batch_col,
        )
        
        # Open H5 file to read expression data
        with h5py.File(h5_path, "r") as f:
            # Read expression matrix - try obsm/X_hvg first, fallback to X
            if self.embed_key and f"obsm/{self.embed_key}" in f:
                X = np.array(f[f"obsm/{self.embed_key}"])
                logger.info(f"Using obsm/{self.embed_key} with shape {X.shape}")
            elif "obsm/X_hvg" in f:
                X = np.array(f["obsm/X_hvg"])
                logger.info(f"Using obsm/X_hvg with shape {X.shape}")
            else:
                X_ds = f["X"]
                attrs = dict(X_ds.attrs) if hasattr(X_ds, 'attrs') else {}
                if attrs.get('encoding-type') == 'csr_matrix':
                    logger.warning("CSR sparse matrix detected - reading may be slow")
                    # For now, let's skip CSR and require dense
                    raise NotImplementedError(
                        "CSR sparse matrix not supported. Please use obsm/X_hvg or convert X to dense."
                    )
                X = np.array(X_ds)
                logger.info(f"Using X with shape {X.shape}")
            
            if X.ndim != 2:
                raise ValueError(f"Expected 2D expression matrix, got shape {X.shape}")
        
        n_cells = X.shape[0]
        logger.info(f"Processing {n_cells} cells from {Path(h5_path).name}")
        
        # Organize cells by (perturbation, batch)
        cells_by_pert_batch: Dict[tuple, List[int]] = {}
        for cell_idx in range(n_cells):
            pert_code = cache.pert_codes[cell_idx]
            pert_name = cache.pert_categories[pert_code]
            batch_code = cache.batch_codes[cell_idx]
            batch_name = cache.batch_categories[batch_code]
            cell_type_code = cache.cell_type_codes[cell_idx]
            cell_type = cache.cell_type_categories[cell_type_code]
            
            key = (pert_name, batch_name, cell_type)
            if key not in cells_by_pert_batch:
                cells_by_pert_batch[key] = []
            cells_by_pert_batch[key].append(cell_idx)
        
        # Get unique perturbations (excluding control)
        unique_perts = set()
        for (pert, batch, ct), indices in cells_by_pert_batch.items():
            if pert != self.control_pert:
                unique_perts.add(pert)
        
        logger.info(f"Found {len(unique_perts)} unique perturbations (excluding control)")
        
        # Create pairs for each perturbation
        n_pairs_created = 0
        for pert_name in unique_perts:
            # Get perturbation embedding
            pert_emb = self._get_pert_embedding(pert_name)
            if pert_emb is None:
                continue
            
            # Find all perturbed cells for this perturbation (across all batches/cell types)
            all_pert_cells = []
            for (p, batch, ct), indices in cells_by_pert_batch.items():
                if p == pert_name:
                    all_pert_cells.extend([(idx, batch, ct) for idx in indices])
            
            if len(all_pert_cells) == 0:
                continue
            
            # Sample perturbed cells
            n_to_sample = min(self.n_pert_samples, len(all_pert_cells))
            sampled_pert_cells = self.rng.choice(
                len(all_pert_cells), size=n_to_sample, replace=False
            )
            
            for sample_idx in sampled_pert_cells:
                pert_cell_idx, batch_name, cell_type = all_pert_cells[sample_idx]
                
                # Find control cells from the same batch and cell type
                ctrl_key = (self.control_pert, batch_name, cell_type)
                if ctrl_key in cells_by_pert_batch:
                    ctrl_cells = cells_by_pert_batch[ctrl_key]
                else:
                    # Fallback: any control cells from same cell type
                    ctrl_cells = []
                    for (p, b, ct), indices in cells_by_pert_batch.items():
                        if p == self.control_pert and ct == cell_type:
                            ctrl_cells.extend(indices)
                
                if len(ctrl_cells) == 0:
                    # Final fallback: any control cells
                    for (p, b, ct), indices in cells_by_pert_batch.items():
                        if p == self.control_pert:
                            ctrl_cells.extend(indices)
                
                if len(ctrl_cells) == 0:
                    logger.warning(f"No control cells found for perturbation '{pert_name}'")
                    continue
                
                # Sample control cells
                n_ctrl = min(self.n_ctrl_samples, len(ctrl_cells))
                sampled_ctrl_indices = self.rng.choice(ctrl_cells, size=n_ctrl, replace=False)
                
                # Create a pair for each control cell
                for ctrl_idx in sampled_ctrl_indices:
                    pair = {
                        'ctrl_cell_emb': torch.tensor(X[ctrl_idx], dtype=torch.float32),
                        'pert_cell_emb': torch.tensor(X[pert_cell_idx], dtype=torch.float32),
                        'pert_embedding': pert_emb,
                        'perturbation': pert_name,
                        'cell_type': cell_type,
                        'batch': batch_name,
                    }
                    self.pairs.append(pair)
                    n_pairs_created += 1
        
        logger.info(f"Created {n_pairs_created} pairs from {Path(h5_path).name}")
    
    def _get_pert_embedding(self, pert_name: str) -> Optional[torch.Tensor]:
        """
        Get perturbation embedding by name.
        Handles case variations and missing embeddings.
        """
        # Try exact match first
        if pert_name in self.pert_embedding_map:
            return self.pert_embedding_map[pert_name]
        
        # Try case variations
        for key in self.pert_embedding_map.keys():
            if key.lower() == pert_name.lower():
                logger.debug(f"Found embedding for '{pert_name}' using case-insensitive match: '{key}'")
                return self.pert_embedding_map[key]
        
        # Try uppercase (common for gene names)
        if pert_name.upper() in self.pert_embedding_map:
            return self.pert_embedding_map[pert_name.upper()]
        
        # Try lowercase
        if pert_name.lower() in self.pert_embedding_map:
            return self.pert_embedding_map[pert_name.lower()]
        
        logger.warning(f"No embedding found for '{pert_name}', skipping")
        return None
    
    def __len__(self) -> int:
        return len(self.pairs)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return self.pairs[idx]


def collate_perturbation_batch(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """
    Collate function for perturbation batches.
    
    Args:
        batch: List of pair dictionaries from PerturbationDataset
        
    Returns:
        Batched dictionary with stacked tensors
    """
    if len(batch) == 0:
        return {}
    
    # Stack all tensors
    collated = {
        'ctrl_cell_emb': torch.stack([item['ctrl_cell_emb'] for item in batch]),
        'pert_cell_emb': torch.stack([item['pert_cell_emb'] for item in batch]),
        'pert_embedding': torch.stack([item['pert_embedding'] for item in batch]),
        'perturbation': [item['perturbation'] for item in batch],
        'cell_type': [item['cell_type'] for item in batch],
        'batch': [item['batch'] for item in batch],
    }
    
    return collated
