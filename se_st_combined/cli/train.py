"""
Training script for SE+ST Combined Model with Hydra configuration
"""

import logging
import sys
from pathlib import Path
from typing import Optional

import hydra
from omegaconf import DictConfig, OmegaConf
import torch
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning.pytorch.loggers import WandbLogger, TensorBoardLogger

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def setup_logger(cfg: DictConfig):
    """Setup experiment logger"""
    loggers = []
    
    # TensorBoard logger (always enabled)
    tb_logger = TensorBoardLogger(
        save_dir=cfg.get('output_dir', './outputs'),
        name=cfg.get('name', 'se_st_training'),
    )
    loggers.append(tb_logger)
    
    # W&B logger (optional)
    if cfg.get('wandb', {}).get('enabled', False):
        wandb_logger = WandbLogger(
            project=cfg.wandb.get('project', 'se-st-combined'),
            entity=cfg.wandb.get('entity', None),
            name=cfg.get('name', 'se_st_training'),
            save_dir=cfg.get('output_dir', './outputs'),
            tags=cfg.wandb.get('tags', []),
        )
        loggers.append(wandb_logger)
        logger.info("Weights & Biases logging enabled")
    
    return loggers


def setup_callbacks(cfg: DictConfig):
    """Setup training callbacks"""
    callbacks = []
    
    output_dir = Path(cfg.get('output_dir', './outputs'))
    experiment_name = cfg.get('name', 'se_st_training')
    
    # Model checkpoint callback
    checkpoint_callback = ModelCheckpoint(
        dirpath=output_dir / experiment_name / "checkpoints",
        filename="step={step}-val_loss={val_loss:.4f}",
        monitor="val_loss",
        mode="min",
        save_top_k=cfg.get('save_top_k', 3),
        save_last=True,
        verbose=True,
        every_n_train_steps=cfg.training.get('ckpt_every_n_steps', 1000),
    )
    callbacks.append(checkpoint_callback)
    
    # Early stopping callback (optional)
    if cfg.training.get('early_stopping', False):
        early_stop_callback = EarlyStopping(
            monitor="val_loss",
            mode="min",
            patience=cfg.training.get('patience', 10),
            verbose=True,
        )
        callbacks.append(early_stop_callback)
    
    return callbacks


@hydra.main(version_base=None, config_path="../configs", config_name="se_st_combined")
def main(cfg: DictConfig):
    """
    Main training function with Hydra configuration.
    
    Usage:
        # Using default config
        se-st-train
        
        # Override config parameters
        se-st-train model.kwargs.st_hidden_dim=1024 training.batch_size=16
        
        # Use different config file
        se-st-train --config-name custom_config
    """
    # Disable struct mode to allow flexible configuration
    OmegaConf.set_struct(cfg, False)
    
    logger.info("=" * 60)
    logger.info("SE+ST Combined Model Training with Hydra")
    logger.info("=" * 60)
    logger.info(f"Configuration:\n{OmegaConf.to_yaml(cfg)}")
    
    # Create output directory
    output_dir = Path(cfg.get('output_dir', './outputs'))
    experiment_name = cfg.get('name', 'se_st_training')
    full_output_dir = output_dir / experiment_name
    full_output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {full_output_dir}")
    
    # Save the full configuration
    config_save_path = full_output_dir / "config.yaml"
    with open(config_save_path, 'w') as f:
        OmegaConf.save(cfg, f)
    logger.info(f"Configuration saved to: {config_save_path}")
    
    # Setup loggers
    loggers = setup_logger(cfg)
    
    # Setup callbacks
    callbacks = setup_callbacks(cfg)
    
    # Initialize trainer
    trainer = Trainer(
        max_steps=cfg.training.get('max_steps', 40000),
        max_epochs=cfg.training.get('max_epochs', None),
        accelerator=cfg.get('accelerator', 'auto'),
        devices=cfg.get('devices', 'auto'),
        logger=loggers,
        callbacks=callbacks,
        log_every_n_steps=cfg.training.get('log_every_n_steps', 100),
        val_check_interval=cfg.training.get('val_check_interval', 1000),
        precision=cfg.training.get('precision', '16-mixed' if torch.cuda.is_available() else '32'),
        gradient_clip_val=cfg.training.get('gradient_clip_val', None),
        accumulate_grad_batches=cfg.training.get('accumulate_grad_batches', 1),
    )
    
    logger.info("Trainer initialized")
    logger.info(f"Device: {cfg.get('accelerator', 'auto')}")
    
    # Note: Model and data loading should be implemented based on your specific needs
    logger.error("Model and data loading not yet implemented!")
    logger.error("Please implement model initialization and data loading in this script.")
    logger.error("")
    logger.error("Example implementation:")
    logger.error("  from se_st_combined.models.state_transition import StateTransitionPerturbationModel")
    logger.error("  from se_st_combined.models.se_st_combined import SE_ST_CombinedModel")
    logger.error("")
    logger.error("  # Initialize model based on config")
    logger.error("  if cfg.model.name == 'se_st_combined':")
    logger.error("      model = SE_ST_CombinedModel(**cfg.model.kwargs)")
    logger.error("  else:")
    logger.error("      model = StateTransitionPerturbationModel(**cfg.model.kwargs)")
    logger.error("")
    logger.error("  # Initialize data module")
    logger.error("  datamodule = YourDataModule(**cfg.data.kwargs)")
    logger.error("")
    logger.error("  # Train")
    logger.error("  trainer.fit(model, datamodule)")
    logger.error("")
    logger.error("For now, please use the examples/simple_train.py script for actual training.")
    
    return 1


if __name__ == "__main__":
    sys.exit(main())

