"""
Debug script to check cell counts per perturbation
"""
import h5py
import numpy as np
from pathlib import Path
from collections import defaultdict

def safe_decode(x):
    if isinstance(x, (bytes, bytearray)):
        return x.decode("utf-8", errors="ignore")
    return str(x)

# Check one H5 file
h5_files = list(Path("/data").glob("*.h5"))
print(f"Found {len(h5_files)} H5 files")

if h5_files:
    h5_file = h5_files[0]
    print(f"\nAnalyzing: {h5_file.name}")
    
    with h5py.File(h5_file, "r") as f:
        # Read perturbation info
        pert_categories = np.array([safe_decode(x) for x in f["obs/target_gene/categories"][:]])
        pert_codes = f["obs/target_gene/codes"][:]
        
        # Read cell type info
        ct_categories = np.array([safe_decode(x) for x in f["obs/cell_type/categories"][:]])
        ct_codes = f["obs/cell_type/codes"][:]
        
        # Count cells per (perturbation, cell_type)
        counts = defaultdict(lambda: defaultdict(int))
        for i in range(len(pert_codes)):
            pert = pert_categories[pert_codes[i]]
            ct = ct_categories[ct_codes[i]]
            counts[pert][ct] += 1
        
        print(f"\nTotal perturbations: {len(counts)}")
        print(f"Total cells: {len(pert_codes)}")
        
        # Check how many have >= 128 cells
        perts_with_128 = []
        perts_by_count = {}
        
        for pert, ct_counts in counts.items():
            total_cells = sum(ct_counts.values())
            perts_by_count[pert] = total_cells
            if total_cells >= 128:
                perts_with_128.append((pert, total_cells, ct_counts))
        
        print(f"\nPerturbations with >= 128 cells: {len(perts_with_128)}")
        print(f"Perturbations with < 128 cells: {len(counts) - len(perts_with_128)}")
        
        # Show distribution
        count_ranges = {
            "1-10": 0,
            "11-50": 0,
            "51-100": 0,
            "101-127": 0,
            "128+": 0,
        }
        
        for count in perts_by_count.values():
            if count <= 10:
                count_ranges["1-10"] += 1
            elif count <= 50:
                count_ranges["11-50"] += 1
            elif count <= 100:
                count_ranges["51-100"] += 1
            elif count <= 127:
                count_ranges["101-127"] += 1
            else:
                count_ranges["128+"] += 1
        
        print(f"\nDistribution of cell counts:")
        for range_name, num_perts in count_ranges.items():
            print(f"  {range_name} cells: {num_perts} perturbations")
        
        # Show examples with >= 128 cells
        if perts_with_128:
            print(f"\nExamples with >= 128 cells:")
            for pert, total, ct_counts in sorted(perts_with_128, key=lambda x: -x[1])[:10]:
                print(f"  {pert}: {total} cells across {len(ct_counts)} cell types")
                for ct, count in sorted(ct_counts.items(), key=lambda x: -x[1])[:3]:
                    print(f"    - {ct}: {count} cells")
        
        # Check control cells
        control_pert = "non-targeting"
        if control_pert in counts:
            print(f"\n'{control_pert}' (control):")
            total_ctrl = sum(counts[control_pert].values())
            print(f"  Total: {total_ctrl} cells")
            for ct, count in sorted(counts[control_pert].items(), key=lambda x: -x[1]):
                print(f"    - {ct}: {count} cells")

