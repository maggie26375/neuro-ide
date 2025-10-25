"""
Training script for Neural ODE-based perturbation prediction model.

This script provides training functionality for the integrated Neural ODE system.
"""

import torch
import torch.nn as nn
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning.pytorch.loggers import TensorBoardLogger
import hydra
from omegaconf import DictConfig
import logging
from typing import Tuple
import sys
import os

# Add the parent directory to the path to fix imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.se_st_combined_neural_ode import SE_ST_NeuralODE_Model
from se_st_combined.cli.train import SE_ST_DataModule

logger = logging.getLogger(__name__)

def train_neural_ode(
    data_dir: str = "/data",
    batch_size: int = 16,
    max_epochs: int = 100,
    lr: float = 1e-4,
    use_neural_ode: bool = True,
    ode_hidden_dim: int = 128,
    ode_layers: int = 3,
    time_range: Tuple[float, float] = (0.0, 1.0),
    num_time_points: int = 10,
    se_model_path: str = "SE-600M",
    se_checkpoint_path: str = "SE-600M/se600m_epoch15.ckpt"
) -> SE_ST_NeuralODE_Model:
    """
    训练 Neural ODE 模型
    
    Args:
        data_dir: 数据目录
        batch_size: 批次大小
        max_epochs: 最大训练轮数
        lr: 学习率
        use_neural_ode: 是否使用 Neural ODE
        ode_hidden_dim: ODE 隐藏层维度
        ode_layers: ODE 层数
        time_range: 时间范围
        num_time_points: 时间点数
        se_model_path: SE 模型路径
        se_checkpoint_path: SE 检查点路径
    
    Returns:
        训练好的模型
    """
    
    # 创建模型
    model = SE_ST_NeuralODE_Model(
        input_dim=18080,
        hidden_dim=512,
        output_dim=512,
        pert_dim=1280,
        se_model_path=se_model_path,
        se_checkpoint_path=se_checkpoint_path,
        freeze_se_model=True,
        st_hidden_dim=512,
        st_cell_set_len=128,
        use_neural_ode=use_neural_ode,
        ode_hidden_dim=ode_hidden_dim,
        ode_layers=ode_layers,
        time_range=time_range,
        num_time_points=num_time_points,
        lr=lr
    )
    
    # 创建数据模块
    datamodule = SE_ST_DataModule(
        toml_config_path="/data/neural_ode_starter.toml",
        perturbation_features_file="/data/ESM2_pert_features.pt",
        batch_size=batch_size,
        num_workers=4
    )
    
    # 设置回调
    callbacks = setup_callbacks()
    
    # 设置日志记录器
    logger_tb = TensorBoardLogger(
        save_dir="logs",
        name="neural_ode_experiment"
    )
    
    # 训练器配置
    trainer = Trainer(
        max_epochs=max_epochs,
        callbacks=callbacks,
        logger=logger_tb,
        val_check_interval=0.5,
        gradient_clip_val=1.0,
        precision=16,
        devices="auto",  # 自动检测可用设备
        accelerator="gpu"
    )
    
    # 训练
    trainer.fit(model, datamodule)
    
    return model

def setup_callbacks():
    """设置训练回调"""
    callbacks = []
    
    # 模型检查点
    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints",
        filename="neural_ode-{epoch:02d}-{val_loss:.2f}",
        save_top_k=3,
        monitor="val_loss",
        mode="min",
        save_last=False,
        save_on_train_epoch_end=False
    )
    callbacks.append(checkpoint_callback)
    
    # 早停
    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=10,
        mode="min"
    )
    callbacks.append(early_stopping)
    
    return callbacks

@hydra.main(version_base=None, config_path="../configs", config_name="neural_ode_config")
def main(cfg: DictConfig) -> None:
    """主训练函数"""
    
    # 设置日志
    logging.basicConfig(level=logging.INFO)
    
    # 训练模型
    model = train_neural_ode(
        data_dir=cfg.data.data_dir,
        batch_size=cfg.data.batch_size,
        max_epochs=cfg.training.max_epochs,
        lr=cfg.training.optimizer.lr,
        use_neural_ode=cfg.model.use_neural_ode,
        ode_hidden_dim=cfg.model.ode_hidden_dim,
        ode_layers=cfg.model.ode_layers,
        time_range=tuple(cfg.model.time_range),
        num_time_points=cfg.model.num_time_points,
        se_model_path=cfg.model.se_model_path,
        se_checkpoint_path=cfg.model.se_checkpoint_path
    )
    
    logger.info("Training completed successfully!")
    
    return model

if __name__ == "__main__":
    main()
