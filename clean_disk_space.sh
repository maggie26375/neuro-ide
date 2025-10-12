#!/bin/bash
# Disk Space Cleanup Script for SE+ST Combined Training
# Use this when you run out of disk space during inference

set -e

echo "🧹 SE+ST Combined - Disk Space Cleanup"
echo "========================================"
echo ""

# Function to show disk usage
show_disk_usage() {
    echo "💾 Current Disk Usage:"
    df -h | grep -E "Filesystem|/dev/md0p1"
    echo ""
}

# Show current usage
show_disk_usage

# 1. Clean pip cache
echo "🗑️  Step 1: Cleaning pip cache..."
pip cache purge 2>/dev/null || true
echo "✅ Pip cache cleaned"
echo ""

# 2. Clean torch cache
echo "🗑️  Step 2: Cleaning PyTorch cache..."
rm -rf ~/.cache/torch 2>/dev/null || true
echo "✅ PyTorch cache cleaned"
echo ""

# 3. Clean old checkpoints (keep only top-2 and final)
echo "🗑️  Step 3: Cleaning old checkpoints..."
if [ -d "competition" ]; then
    # Remove all checkpoints except the newest 2
    find competition -name "*.ckpt" -type f | 
    grep -v "final_model.ckpt" | 
    sort -r | 
    tail -n +3 | 
    while read f; do
        echo "  Removing: $f ($(du -h "$f" | cut -f1))"
        rm -f "$f"
    done
    echo "✅ Old checkpoints cleaned"
else
    echo "  No competition directory found"
fi
echo ""

# 4. Clean old predictions
echo "🗑️  Step 4: Cleaning old prediction files..."
if [ -d "competition" ]; then
    find competition -name "prediction*.h5ad" -type f | 
    while read f; do
        echo "  Removing: $f ($(du -h "$f" | cut -f1))"
        rm -f "$f"
    done
    echo "✅ Old predictions cleaned"
else
    echo "  No predictions found"
fi
echo ""

# 5. Clean wandb artifacts (optional - comment out if you want to keep)
# echo "🗑️  Step 5: Cleaning wandb artifacts..."
# rm -rf wandb 2>/dev/null || true
# echo "✅ Wandb artifacts cleaned"
# echo ""

# 6. Clean Python __pycache__
echo "🗑️  Step 5: Cleaning Python cache..."
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
find . -type f -name "*.pyc" -delete 2>/dev/null || true
echo "✅ Python cache cleaned"
echo ""

# 7. Clean temporary files
echo "🗑️  Step 6: Cleaning temporary files..."
rm -rf /tmp/* 2>/dev/null || true
echo "✅ Temporary files cleaned"
echo ""

# Show final usage
show_disk_usage

echo "✅ Cleanup complete!"
echo ""
echo "💡 If you still need more space, you can:"
echo "   1. Delete old training runs: rm -rf competition/old_run_name"
echo "   2. Delete SE model if not needed: rm -rf SE-600M"
echo "   3. Compress predictions: gzip competition/prediction.h5ad"
echo ""

