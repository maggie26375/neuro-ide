#!/bin/bash

# Neural ODE Training Script
# This script runs the Neural ODE training with proper Hydra configuration
# Note: config_path and config_name are already set in the @hydra.main decorator

cd /Users/maggie/Desktop/contrast/se_st_combined_package

python -m neuro-IDE.cli.train_neural_ode \
  data.toml_config_path="/data/neural_ode_starter.toml" \
  data.perturbation_features_file="/data/ESM2_pert_features.pt" \
  data.num_workers=4 \
  training.max_steps=80000 \
  training.batch_size=8 \
  training.val_check_interval=100 \
  model.input_dim=18080 \
  model.use_neural_ode=true \
  model.ode_hidden_dim=128 \
  model.ode_layers=3 \
  model.time_range=[0.0,1.0] \
  model.num_time_points=10
