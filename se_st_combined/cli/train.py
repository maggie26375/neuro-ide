"""
Training script for SE+ST Combined Model
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

import torch
import yaml
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning.pytorch.loggers import WandbLogger, TensorBoardLogger

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Train SE+ST Combined Model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Model arguments
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to configuration YAML file"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="se_st_combined",
        choices=["se_st_combined", "state_transition"],
        help="Model type to train"
    )
    
    # Data arguments
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="Path to training data"
    )
    parser.add_argument(
        "--val_data_path",
        type=str,
        default=None,
        help="Path to validation data"
    )
    
    # Training arguments
    parser.add_argument(
        "--max_epochs",
        type=int,
        default=100,
        help="Maximum number of training epochs"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for training"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Learning rate"
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=8,
        help="Number of data loading workers"
    )
    
    # Model hyperparameters
    parser.add_argument(
        "--hidden_dim",
        type=int,
        default=512,
        help="Hidden dimension size"
    )
    parser.add_argument(
        "--cell_set_len",
        type=int,
        default=128,
        help="Cell set length for transformer"
    )
    
    # Output arguments
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./outputs",
        help="Output directory for checkpoints and logs"
    )
    parser.add_argument(
        "--name",
        type=str,
        default="se_st_training",
        help="Experiment name"
    )
    
    # Logging arguments
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="Use Weights & Biases for logging"
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="se-st-combined",
        help="W&B project name"
    )
    parser.add_argument(
        "--wandb_entity",
        type=str,
        default=None,
        help="W&B entity (username or team)"
    )
    
    # Hardware arguments
    parser.add_argument(
        "--gpus",
        type=int,
        default=1,
        help="Number of GPUs to use"
    )
    parser.add_argument(
        "--accelerator",
        type=str,
        default="auto",
        choices=["auto", "cpu", "gpu", "tpu", "ipu", "hpu"],
        help="Accelerator type"
    )
    
    # Checkpoint arguments
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint to resume from"
    )
    parser.add_argument(
        "--save_top_k",
        type=int,
        default=3,
        help="Save top k checkpoints"
    )
    
    return parser.parse_args()


def load_config(config_path: Optional[str] = None) -> dict:
    """Load configuration from YAML file"""
    if config_path is None:
        return {}
    
    config_path = Path(config_path)
    if not config_path.exists():
        logger.warning(f"Config file not found: {config_path}")
        return {}
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    logger.info(f"Loaded configuration from {config_path}")
    return config


def setup_logger(args):
    """Setup experiment logger"""
    loggers = []
    
    # TensorBoard logger (always enabled)
    tb_logger = TensorBoardLogger(
        save_dir=args.output_dir,
        name=args.name,
    )
    loggers.append(tb_logger)
    
    # W&B logger (optional)
    if args.wandb:
        wandb_logger = WandbLogger(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.name,
            save_dir=args.output_dir,
        )
        loggers.append(wandb_logger)
        logger.info("Weights & Biases logging enabled")
    
    return loggers


def setup_callbacks(args):
    """Setup training callbacks"""
    callbacks = []
    
    # Model checkpoint callback
    checkpoint_callback = ModelCheckpoint(
        dirpath=Path(args.output_dir) / args.name / "checkpoints",
        filename="{epoch:02d}-{val_loss:.4f}",
        monitor="val_loss",
        mode="min",
        save_top_k=args.save_top_k,
        save_last=True,
        verbose=True,
    )
    callbacks.append(checkpoint_callback)
    
    # Early stopping callback
    early_stop_callback = EarlyStopping(
        monitor="val_loss",
        mode="min",
        patience=10,
        verbose=True,
    )
    callbacks.append(early_stop_callback)
    
    return callbacks


def main():
    """Main training function"""
    # Parse arguments
    args = parse_args()
    
    logger.info("=" * 60)
    logger.info("SE+ST Combined Model Training")
    logger.info("=" * 60)
    logger.info(f"Arguments: {vars(args)}")
    
    # Load configuration
    config = load_config(args.config)
    
    # Create output directory
    output_dir = Path(args.output_dir) / args.name
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {output_dir}")
    
    # Setup loggers
    loggers = setup_logger(args)
    
    # Setup callbacks
    callbacks = setup_callbacks(args)
    
    # Initialize trainer
    trainer = Trainer(
        max_epochs=args.max_epochs,
        accelerator=args.accelerator,
        devices=args.gpus if args.accelerator == "gpu" else "auto",
        logger=loggers,
        callbacks=callbacks,
        log_every_n_steps=10,
        precision="16-mixed" if torch.cuda.is_available() else "32",
    )
    
    logger.info("Trainer initialized")
    logger.info(f"Device: {args.accelerator}")
    logger.info(f"Number of devices: {args.gpus if args.accelerator == 'gpu' else 'auto'}")
    
    # Note: Model and data loading should be implemented based on your specific needs
    logger.error("Model and data loading not yet implemented!")
    logger.error("Please implement model initialization and data loading in this script.")
    logger.error("Example:")
    logger.error("  from se_st_combined.models.state_transition import StateTransitionPerturbationModel")
    logger.error("  from se_st_combined.data import YourDataModule")
    logger.error("  model = StateTransitionPerturbationModel(...)")
    logger.error("  datamodule = YourDataModule(...)")
    logger.error("  trainer.fit(model, datamodule)")
    
    return 1


if __name__ == "__main__":
    sys.exit(main())

