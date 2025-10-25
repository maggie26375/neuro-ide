# Neural ODE Training Guide

## Fixed Issues

### 1. Ambiguous --config option error

**Problem**: The Hydra framework was receiving an ambiguous `--config` argument that could match `--config-path`, `--config-name`, or `--config-dir`.

**Solution**: Removed the `--config` flag from the command line. The configuration is now specified in the `@hydra.main` decorator in the training script:

```python
@hydra.main(version_base=None, config_path="../configs", config_name="neural_ode_config")
```

### 2. Import errors

**Problem**: The training script had import errors due to:
- Incorrect relative imports
- Missing backward compatibility for different pytorch-lightning versions

**Solution**:
- Added try/except blocks for lightning imports to support both old and new package structures
- Fixed path imports by adding parent directory to sys.path
- Changed from relative imports to absolute imports

## Environment Setup

Before running the training script, ensure you have the required packages installed:

```bash
# Install pytorch-lightning (if not already installed)
conda install pytorch-lightning -c conda-forge

# OR using pip
pip install pytorch-lightning

# Also make sure you have:
pip install hydra-core omegaconf torchdiffeq
```

## Running the Training

### Option 1: Using the shell script

```bash
cd /Users/maggie/Desktop/contrast/se_st_combined_package
bash neuro-IDE/run_neural_ode_training.sh
```

### Option 2: Direct Python command

```bash
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
```

### Option 3: Override config file location (if needed)

If you need to use a different config file:

```bash
python -m neuro-IDE.cli.train_neural_ode \
  --config-name=your_custom_config \
  data.toml_config_path="/path/to/your/data.toml"
```

## Configuration

The training configuration is stored in `neuro-IDE/configs/neural_ode_config.yaml`. You can:

1. Edit this file directly
2. Override specific parameters via command line (as shown above)
3. Create a new config file and use `--config-name` to specify it

## Files Modified

1. `neuro-IDE/cli/train_neural_ode.py` - Fixed imports and path handling
2. `neuro-IDE/cli/train_neural_ode_fixed.py` - Fixed imports and path handling
3. `neuro-IDE/run_neural_ode_training.sh` - Created proper training script without ambiguous flags
4. `neuro-IDE/README_TRAINING.md` - This documentation file

## Next Steps

After fixing any remaining environment-specific issues, you should be able to:

1. Run the training script
2. Monitor training progress in logs/
3. Find checkpoints in checkpoints/
4. View tensorboard logs: `tensorboard --logdir=logs`
