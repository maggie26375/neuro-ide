"""
Check where the actual H5 files are located
"""
import os
from pathlib import Path

print("=" * 80)
print("Checking data locations...")
print("=" * 80)

# Check if data directory exists
data_paths_to_check = [
    "/data",
    "/content/state",
    "./data",
    "data",
]

for path in data_paths_to_check:
    p = Path(path)
    if p.exists():
        print(f"\n✅ {path} EXISTS")
        if p.is_dir():
            # List subdirectories
            try:
                subdirs = [d.name for d in p.iterdir() if d.is_dir()]
                print(f"   Subdirectories: {subdirs[:10]}")
                
                # Look for H5 files
                h5_files = list(p.glob("**/*.h5"))
                h5ad_files = list(p.glob("**/*.h5ad"))
                print(f"   H5 files found: {len(h5_files)}")
                print(f"   H5AD files found: {len(h5ad_files)}")
                
                if h5_files:
                    print(f"   Sample H5 files:")
                    for f in h5_files[:5]:
                        print(f"     - {f}")
            except PermissionError:
                print(f"   Permission denied")
    else:
        print(f"\n❌ {path} DOES NOT EXIST")

# Check for competition_support_set specifically
competition_paths = [
    "/data/competition_support_set",
    "./data/competition_support_set",
    "data/competition_support_set",
]

print("\n" + "=" * 80)
print("Checking competition_support_set locations...")
print("=" * 80)

for path in competition_paths:
    p = Path(path)
    if p.exists():
        print(f"\n✅ {path} EXISTS")
        h5_files = sorted(p.glob("*.h5"))
        print(f"   H5 files: {len(h5_files)}")
        for f in h5_files:
            print(f"     - {f.name}")
    else:
        print(f"\n❌ {path} DOES NOT EXIST")

