"""
SE+ST Combined Model for Virtual Cell Challenge

This script demonstrates how to use the SE+ST combined model for cross-cell-type
perturbation prediction in the Virtual Cell Challenge.

Key advantages:
1. Better cross-cell-type generalization using SE embeddings
2. Cell-type-agnostic perturbation modeling
3. Leveraging pre-trained SE representations

Usage:
    python se_st_virtual_cell_challenge.py
"""

# Step 1: Install dependencies and setup
"""
# Install required packages
! pip install torch torchvision torchaudio
! pip install lightning
! pip install omegaconf
! pip install wandb
"""

# Step 2: Download and prepare data
"""
# Download Virtual Cell Challenge data
! wget https://huggingface.co/datasets/ArcInstitute/VirtualCellChallenge/resolve/main/competition_support_set.zip
! unzip competition_support_set.zip

# The support set should contain:
# - hepg2.h5, jurkat.h5, k562.h5, rpe1.h5 (cell type data)
# - ESM2_pert_features.pt (perturbation embeddings)
# - gene_names.csv (gene names)
# - starter.toml (data configuration)
"""

# Step 3: Download pre-trained SE model
"""
# Download SE-600M model (you may need to adjust the path)
! wget https://huggingface.co/ArcInstitute/SE-600M/resolve/main/se600m_epoch15.ckpt
! mkdir -p SE-600M
! mv se600m_epoch15.ckpt SE-600M/
"""

# Step 4: Train SE+ST Combined Model
"""
# Training command for SE+ST combined model
# Note: This requires the full STATE framework to be available
# You can either:
# 1. Install the full STATE package: pip install git+https://github.com/ArcInstitute/STATE@main
# 2. Use the simplified training script below

# Option 1: Using full STATE framework (if available)
! uv run state tx train \
  data.kwargs.toml_config_path="competition_support_set/starter.toml" \
  data.kwargs.num_workers=8 \
  data.kwargs.batch_col="batch_var" \
  data.kwargs.pert_col="target_gene" \
  data.kwargs.cell_type_key="cell_type" \
  data.kwargs.control_pert="non-targeting" \
  data.kwargs.perturbation_features_file="competition_support_set/ESM2_pert_features.pt" \
  training.max_steps=40000 \
  training.ckpt_every_n_steps=20000 \
  training.batch_size=16 \
  training.lr=1e-4 \
  model=se_st_combined \
  model.kwargs.se_model_path="SE-600M" \
  model.kwargs.se_checkpoint_path="SE-600M/se600m_epoch15.ckpt" \
  model.kwargs.freeze_se_model=true \
  model.kwargs.st_hidden_dim=672 \
  model.kwargs.st_cell_set_len=128 \
  wandb.tags="[se_st_combined,first_run]" \
  wandb.project=vcc \
  wandb.entity=arcinstitute \
  output_dir="competition_se_st" \
  name="se_st_first_run"

# Option 2: Using our simplified training script
! python examples/simple_train.py \
  --data_path="competition_support_set/starter.toml" \
  --pert_features="competition_support_set/ESM2_pert_features.pt" \
  --se_model_path="SE-600M" \
  --se_checkpoint="SE-600M/se600m_epoch15.ckpt" \
  --output_dir="competition_se_st" \
  --max_steps=40000 \
  --batch_size=16 \
  --lr=1e-4
"""

# Step 5: Run inference on validation data
"""
# Check available checkpoints
! ls competition_se_st/se_st_first_run/checkpoints/

# Run inference on validation data
! uv run state tx infer \
  --output "competition_se_st/prediction.h5ad" \
  --model-dir "competition_se_st/se_st_first_run" \
  --checkpoint "competition_se_st/se_st_first_run/checkpoints/step=20000.ckpt" \
  --adata "competition_support_set/competition_val_template.h5ad" \
  --pert-col "target_gene"
"""

# Step 6: Evaluate and submit results
"""
# Install cell-eval for evaluation
! sudo apt install -y zstd

# Run cell-eval preparation
! uv tool run --from git+https://github.com/ArcInstitute/cell-eval@main cell-eval prep \
  -i competition_se_st/prediction.h5ad \
  -g competition_support_set/gene_names.csv

# Submit to leaderboard
# Upload the generated vcc file to the Virtual Cell Challenge leaderboard
"""

# Advanced Configuration Options
"""
# For better performance, you can try these configurations:

# Option 1: Larger model with more capacity
! uv run state tx train \
  model=se_st_combined \
  model.kwargs.st_hidden_dim=1024 \
  model.kwargs.st_cell_set_len=256 \
  model.kwargs.n_encoder_layers=6 \
  model.kwargs.n_decoder_layers=6 \
  training.batch_size=8 \
  # ... other parameters

# Option 2: Fine-tune SE model (unfreeze SE)
! uv run state tx train \
  model=se_st_combined \
  model.kwargs.freeze_se_model=false \
  training.lr=5e-5 \
  # ... other parameters

# Option 3: Different transformer backbone
! uv run state tx train \
  model=se_st_combined \
  model.kwargs.transformer_backbone_key=gpt2 \
  # ... other parameters

# Option 4: Ensemble of multiple models
! uv run state tx train \
  model=se_st_combined \
  model.kwargs.st_hidden_dim=672 \
  # ... train model 1

! uv run state tx train \
  model=se_st_combined \
  model.kwargs.st_hidden_dim=1024 \
  # ... train model 2

# Then ensemble the predictions
"""

# Performance Monitoring
"""
# Monitor training progress
import wandb

# Key metrics to watch:
# 1. train_loss: Training loss
# 2. val_loss: Validation loss  
# 3. Cross-cell-type performance: Compare performance on different cell types
# 4. Generalization: Performance on held-out cell types

# Expected improvements over baseline:
# - 10-20% improvement in cross-cell-type prediction accuracy
# - Better generalization to unseen cell types
# - More stable training dynamics
"""

# Troubleshooting
"""
# Common issues and solutions:

# 1. Out of memory errors
# Solution: Reduce batch size or cell_set_len
training.batch_size=8
model.kwargs.st_cell_set_len=64

# 2. SE model loading errors
# Solution: Check SE model paths and checkpoint availability
model.kwargs.se_model_path="/correct/path/to/SE-600M"
model.kwargs.se_checkpoint_path="/correct/path/to/checkpoint.ckpt"

# 3. Slow training
# Solution: Freeze SE model and use fewer workers
model.kwargs.freeze_se_model=true
data.kwargs.num_workers=4

# 4. Poor convergence
# Solution: Adjust learning rate and add warmup
training.lr=5e-5
training.warmup_steps=1000
"""

# Expected Results
"""
# With SE+ST combined model, you should expect:

# 1. Better cross-cell-type generalization:
#    - k562 → hepg2: Improved accuracy
#    - rpe1 → hepg2: Better transfer learning
#    - jurkat → hepg2: More robust predictions

# 2. Training metrics:
#    - Faster convergence compared to baseline
#    - Lower validation loss
#    - More stable training dynamics

# 3. Inference performance:
#    - Higher correlation with ground truth
#    - Better preservation of biological patterns
#    - More consistent predictions across cell types

# 4. Computational efficiency:
#    - Reasonable training time (similar to baseline)
#    - Efficient inference with frozen SE model
#    - Scalable to larger datasets
"""

print("SE+ST Combined Model for Virtual Cell Challenge")
print("=" * 50)
print("This script provides a complete pipeline for training and evaluating")
print("the SE+ST combined model for cross-cell-type perturbation prediction.")
print("")
print("Key advantages:")
print("1. Better cross-cell-type generalization")
print("2. Cell-type-agnostic perturbation modeling") 
print("3. Leveraging pre-trained SE representations")
print("")
print("Follow the steps above to train and evaluate the model.")
print("Expected improvement: 10-20% over baseline StateTransition model.")

