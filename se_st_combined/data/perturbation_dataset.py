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
        cell_sentence_len: int = 128,  # Number of cells per sentence (STATE default)
        random_seed: int = 42,
        embed_key: str = None,  # Not used for now, but kept for compatibility
        **kwargs,  # For backward compatibility (n_ctrl_samples, n_pert_samples)
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
            cell_sentence_len: Number of cells per sentence (STATE default: 128)
            random_seed: Random seed for reproducibility
            embed_key: Key under obsm for embeddings (not used yet)
            **kwargs: Backward compatibility (n_ctrl_samples, n_pert_samples ignored)
        """
        super().__init__()
        self.toml_config_path = Path(toml_config_path)
        self.perturbation_features_file = Path(perturbation_features_file)
        self.split = split
        self.batch_col = batch_col
        self.pert_col = pert_col
        self.cell_type_key = cell_type_key
        self.control_pert = control_pert
        self.cell_sentence_len = cell_sentence_len
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
            # Check if this dataset should be loaded for current split
            training_config = self.config.get("training", {})
            dataset_split = training_config.get(dataset_name, "train")
            
            # For now, always load the dataset if it's marked as "train" in training config
            # We'll handle zeroshot splits at the cell type level inside _load_and_pair_single_h5
            if dataset_split == "train":
                logger.info(f"Loading dataset: {dataset_name} for split: {self.split}")
                
                # Find H5 files
                files = self._find_dataset_files(dataset_path)
                logger.info(f"Found {len(files)} H5 files for {dataset_name}")
                
                for h5_file in files:
                    logger.info(f"Processing {h5_file.name}...")
                    try:
                        self._load_and_pair_single_h5(str(h5_file), dataset_name)
                    except Exception as e:
                        logger.error(f"Error processing {h5_file}: {e}")
                        import traceback
                        traceback.print_exc()
            else:
                logger.debug(f"Skipping dataset {dataset_name} (configured for split: {dataset_split})")
    
    def _load_and_pair_single_h5(self, h5_path: str, dataset_name: str):
        """
        Load a single H5 file and create control/perturbation pairs.
        Now also checks zeroshot configuration to filter cells by split.
        """
        # Load metadata
        cache = H5MetadataCache(
            h5_path,
            pert_col=self.pert_col,
            cell_type_key=self.cell_type_key,
            control_pert=self.control_pert,
            batch_col=self.batch_col,
        )
        
        # Determine which data to use and get shape (but DON'T load full matrix yet!)
        with h5py.File(h5_path, "r") as f:
            if self.embed_key and f"obsm/{self.embed_key}" in f:
                data_key = f"obsm/{self.embed_key}"
                n_cells, n_genes = f[data_key].shape
                logger.info(f"Will use {data_key} with shape ({n_cells}, {n_genes})")
            elif "obsm/X_hvg" in f:
                data_key = "obsm/X_hvg"
                n_cells, n_genes = f[data_key].shape
                logger.info(f"Will use {data_key} with shape ({n_cells}, {n_genes})")
            else:
                X_ds = f["X"]
                attrs = dict(X_ds.attrs) if hasattr(X_ds, 'attrs') else {}
                if attrs.get('encoding-type') == 'csr_matrix':
                    logger.warning(f"Skipping {Path(h5_path).name}: CSR sparse matrix not supported")
                    return  # Skip this file
                data_key = "X"
                n_cells, n_genes = X_ds.shape
                logger.info(f"Will use X with shape ({n_cells}, {n_genes})")
        
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
        
        # Get zeroshot configuration
        zeroshot_config = self.config.get("zeroshot", {})
        
        # Build a map of cell type to split
        celltype_splits = {}
        for key, split_val in zeroshot_config.items():
            # key format: "dataset_name.celltype"
            if "." in key:
                parts = key.split(".", 1)
                if parts[0] == dataset_name:
                    celltype_splits[parts[1]] = split_val
        
        logger.info(f"Zeroshot cell type splits: {celltype_splits}")
        
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
                    # Check if this cell type should be included in current split
                    ct_split = celltype_splits.get(ct, "train")  # default to train
                    if ct_split == self.split:
                        all_pert_cells.extend([(idx, batch, ct) for idx in indices])
            
            if len(all_pert_cells) == 0:
                continue
            
            # Note: We can still create sentences even with fewer cells by sampling with replacement
            # This matches STATE's behavior
            
            # Group perturbed cells by (batch, cell_type) to find matching controls
            pert_cells_by_context = {}
            for idx, batch, ct in all_pert_cells:
                key = (batch, ct)
                if key not in pert_cells_by_context:
                    pert_cells_by_context[key] = []
                pert_cells_by_context[key].append(idx)
            
            # Try to create a sentence from cells with the same batch/cell type
            for (batch_name, cell_type), pert_indices in pert_cells_by_context.items():
                # Check split
                ct_split = celltype_splits.get(cell_type, "train")
                if ct_split != self.split:
                    continue
                
                # Find corresponding control cells
                ctrl_key = (self.control_pert, batch_name, cell_type)
                ctrl_cells = cells_by_pert_batch.get(ctrl_key, [])
                
                # Need at least 1 cell for both pert and ctrl
                if len(pert_indices) == 0 or len(ctrl_cells) == 0:
                    continue
                
                # Sample cell_sentence_len cells for both pert and ctrl
                # Use replacement if not enough cells (matching STATE's behavior)
                replace_pert = len(pert_indices) < self.cell_sentence_len
                replace_ctrl = len(ctrl_cells) < self.cell_sentence_len
                
                sampled_pert_indices = self.rng.choice(
                    pert_indices, size=self.cell_sentence_len, replace=replace_pert
                )
                sampled_ctrl_indices = self.rng.choice(
                    ctrl_cells, size=self.cell_sentence_len, replace=replace_ctrl
                )
                
                # Read only the sampled cells from H5 (memory efficient!)
                with h5py.File(h5_path, "r") as f:
                    ctrl_cell_embeddings = torch.tensor(
                        f[data_key][sampled_ctrl_indices], dtype=torch.float32
                    )  # Shape: [cell_sentence_len, gene_dim]
                    
                    pert_cell_embeddings = torch.tensor(
                        f[data_key][sampled_pert_indices], dtype=torch.float32
                    )  # Shape: [cell_sentence_len, gene_dim]
                
                # Repeat perturbation embedding for each cell in the sentence
                pert_embeddings = pert_emb.unsqueeze(0).repeat(self.cell_sentence_len, 1)
                # Shape: [cell_sentence_len, pert_dim]
                
                sentence = {
                    'ctrl_cell_emb': ctrl_cell_embeddings,
                    'pert_cell_emb': pert_cell_embeddings,
                    'pert_embedding': pert_embeddings,
                    'perturbation': pert_name,
                    'cell_type': cell_type,
                    'batch': batch_name,
                }
                self.pairs.append(sentence)
                n_pairs_created += 1
                
                # Create multiple sentences if we have enough cells
                # Each perturbation/context can create multiple non-overlapping sentences
                # Calculate how many additional sentences we can create
                remaining_pert = len(pert_indices) - self.cell_sentence_len
                remaining_ctrl = len(ctrl_cells) - self.cell_sentence_len
                
                # Create additional sentences (up to 10 per perturbation/context to avoid explosion)
                max_additional = min(10, remaining_pert // self.cell_sentence_len, remaining_ctrl // self.cell_sentence_len)
                for _ in range(max_additional):
                    # Sample different cells
                    sampled_pert_indices = self.rng.choice(
                        pert_indices, size=self.cell_sentence_len, replace=replace_pert
                    )
                    sampled_ctrl_indices = self.rng.choice(
                        ctrl_cells, size=self.cell_sentence_len, replace=replace_ctrl
                    )
                    
                    # Read only the sampled cells from H5 (memory efficient!)
                    with h5py.File(h5_path, "r") as f:
                        ctrl_cell_embeddings = torch.tensor(
                            f[data_key][sampled_ctrl_indices], dtype=torch.float32
                        )
                        pert_cell_embeddings = torch.tensor(
                            f[data_key][sampled_pert_indices], dtype=torch.float32
                        )
                    
                    pert_embeddings = pert_emb.unsqueeze(0).repeat(self.cell_sentence_len, 1)
                    
                    sentence = {
                        'ctrl_cell_emb': ctrl_cell_embeddings,
                        'pert_cell_emb': pert_cell_embeddings,
                        'pert_embedding': pert_embeddings,
                        'perturbation': pert_name,
                        'cell_type': cell_type,
                        'batch': batch_name,
                    }
                    self.pairs.append(sentence)
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
    Collate function for perturbation batches with cell sentences.
    
    Each item in batch already contains [cell_sentence_len, dim] tensors.
    We just need to stack them into [batch_size, cell_sentence_len, dim].
    
    Args:
        batch: List of sentence dictionaries from PerturbationDataset
        
    Returns:
        Batched dictionary with stacked tensors
    """
    if len(batch) == 0:
        return {}
    
    # Stack all sentence tensors
    # Each item['ctrl_cell_emb'] has shape [cell_sentence_len, gene_dim]
    # After stacking: [batch_size, cell_sentence_len, gene_dim]
    collated = {
        'ctrl_cell_emb': torch.stack([item['ctrl_cell_emb'] for item in batch]),
        'pert_cell_emb': torch.stack([item['pert_cell_emb'] for item in batch]),
        'pert_emb': torch.stack([item['pert_embedding'] for item in batch]),
        'perturbation': [item['perturbation'] for item in batch],
        'cell_type': [item['cell_type'] for item in batch],
        'batch': [item['batch'] for item in batch],
    }
    
    return collated
