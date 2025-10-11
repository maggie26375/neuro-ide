"""
Debug script to trace data loading issues
"""
import logging
import sys
from pathlib import Path

# Set up detailed logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(levelname)s [%(name)s] %(message)s',
    stream=sys.stdout
)

import tomli
import torch
import h5py
import numpy as np

print("=" * 80)
print("STEP 1: Check TOML config")
print("=" * 80)

toml_path = "data/starter.toml"
with open(toml_path, "rb") as f:
    config = tomli.load(f)

print(f"Config keys: {list(config.keys())}")
print(f"Datasets: {config.get('datasets', {})}")
print(f"Training: {config.get('training', {})}")
print(f"Zeroshot: {config.get('zeroshot', {})}")
print(f"Fewshot: {config.get('fewshot', {})}")

print("\n" + "=" * 80)
print("STEP 2: Check ESM2 features")
print("=" * 80)

esm2_path = "data/ESM2_pert_features.pt"
esm2_dict = torch.load(esm2_path, map_location="cpu")
print(f"Number of embeddings: {len(esm2_dict)}")
print(f"Sample keys (first 10): {list(esm2_dict.keys())[:10]}")
print(f"Embedding shape: {next(iter(esm2_dict.values())).shape}")

print("\n" + "=" * 80)
print("STEP 3: Check H5 files")
print("=" * 80)

# Get the dataset path from config
for dataset_name, dataset_path in config.get("datasets", {}).items():
    print(f"\nDataset: {dataset_name}")
    print(f"Path pattern: {dataset_path}")
    
    # Expand the pattern
    import glob
    import re
    
    def expand_braces(pattern):
        match = re.search(r"\{([^}]+)\}", pattern)
        if not match:
            return [pattern]
        before = pattern[:match.start()]
        after = pattern[match.end():]
        options = match.group(1).split(",")
        results = []
        for option in options:
            new_text = before + option.strip() + after
            results.extend(expand_braces(new_text))
        return results
    
    expanded = expand_braces(dataset_path)
    print(f"Expanded patterns: {expanded}")
    
    all_files = []
    for pattern in expanded:
        files = sorted(glob.glob(pattern))
        all_files.extend(files)
    
    print(f"Found {len(all_files)} H5 files:")
    for f in all_files:
        print(f"  - {f}")
    
    # Check first H5 file in detail
    if all_files:
        h5_file = all_files[0]
        print(f"\n  Inspecting: {h5_file}")
        
        with h5py.File(h5_file, "r") as f:
            print(f"    H5 groups: {list(f.keys())}")
            
            # Check obs
            if "obs" in f:
                obs = f["obs"]
                print(f"    obs keys: {list(obs.keys())}")
                
                # Check target_gene
                if "target_gene" in obs:
                    tg = obs["target_gene"]
                    print(f"    target_gene type: {type(tg)}")
                    print(f"    target_gene keys: {list(tg.keys()) if hasattr(tg, 'keys') else 'N/A'}")
                    
                    if "categories" in tg:
                        cats = np.array(tg["categories"][:])
                        print(f"    target_gene categories ({len(cats)}): {cats[:10]}")
                        print(f"    categories dtype: {cats.dtype}")
                        
                        # Decode first few
                        decoded = []
                        for c in cats[:10]:
                            if isinstance(c, (bytes, bytearray)):
                                decoded.append(c.decode("utf-8"))
                            else:
                                decoded.append(str(c))
                        print(f"    Decoded categories: {decoded}")
                        
                        # Check if 'non-targeting' is in there
                        all_decoded = []
                        for c in cats:
                            if isinstance(c, (bytes, bytearray)):
                                all_decoded.append(c.decode("utf-8"))
                            else:
                                all_decoded.append(str(c))
                        
                        if "non-targeting" in all_decoded:
                            print(f"    ✅ 'non-targeting' found at index {all_decoded.index('non-targeting')}")
                        else:
                            print(f"    ❌ 'non-targeting' NOT found")
                            # Check for similar
                            similar = [c for c in all_decoded if "target" in c.lower() or "control" in c.lower() or "ctrl" in c.lower()]
                            print(f"    Similar entries: {similar[:10]}")
                    
                    if "codes" in tg:
                        codes = np.array(tg["codes"][:])
                        print(f"    target_gene codes shape: {codes.shape}")
                        print(f"    codes sample: {codes[:20]}")
                        print(f"    unique codes: {len(np.unique(codes))}")
            
            # Check X or obsm
            if "X" in f:
                X = f["X"]
                attrs = dict(X.attrs) if hasattr(X, 'attrs') else {}
                print(f"    X attributes: {attrs}")
                if attrs.get('encoding-type') == 'csr_matrix':
                    print(f"    X is CSR sparse")
                else:
                    print(f"    X shape: {X.shape}")
            
            if "obsm" in f:
                obsm = f["obsm"]
                print(f"    obsm keys: {list(obsm.keys())}")
                if "X_hvg" in obsm:
                    print(f"    X_hvg shape: {obsm['X_hvg'].shape}")

print("\n" + "=" * 80)
print("STEP 4: Try loading with PerturbationDataset")
print("=" * 80)

from se_st_combined.data import PerturbationDataset

dataset = PerturbationDataset(
    toml_config_path="data/starter.toml",
    perturbation_features_file="data/ESM2_pert_features.pt",
    split="train",
    n_ctrl_samples=5,
    n_pert_samples=5,
)

print(f"\nFinal result: {len(dataset)} pairs created")

