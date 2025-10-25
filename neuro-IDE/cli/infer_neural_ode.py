"""
Inference script for Neural ODE-based perturbation prediction model.

This script provides inference functionality for the integrated Neural ODE system.
"""

import torch
import torch.nn as nn
import numpy as np
import scanpy as sc
import anndata as ad
from typing import Dict, Optional, Tuple
import logging

from ..models.se_st_combined_neural_ode import SE_ST_NeuralODE_Model
from ..utils.neural_ode_analysis import analyze_perturbation_dynamics, visualize_perturbation_dynamics

logger = logging.getLogger(__name__)

def load_neural_ode_model(
    checkpoint_path: str,
    device: str = "cuda"
) -> SE_ST_NeuralODE_Model:
    """
    加载训练好的 Neural ODE 模型
    
    Args:
        checkpoint_path: 检查点路径
        device: 设备
    
    Returns:
        加载的模型
    """
    # 加载检查点
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # 获取模型参数
    model_kwargs = checkpoint['hyper_parameters']
    
    # 创建模型
    model = SE_ST_NeuralODE_Model(**model_kwargs)
    
    # 加载权重
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    
    return model

def run_neural_ode_inference(
    model: SE_ST_NeuralODE_Model,
    ctrl_data: ad.AnnData,
    pert_emb: torch.Tensor,
    batch_size: int = 16,
    return_trajectory: bool = False
) -> Tuple[ad.AnnData, Optional[torch.Tensor]]:
    """
    运行 Neural ODE 推理
    
    Args:
        model: 训练好的模型
        ctrl_data: 控制组数据
        pert_emb: 扰动嵌入
        batch_size: 批次大小
        return_trajectory: 是否返回轨迹
    
    Returns:
        predictions: 预测结果
        trajectory: 扰动轨迹（可选）
    """
    model.eval()
    
    with torch.no_grad():
        # 准备数据
        ctrl_expressions = torch.tensor(ctrl_data.X.toarray(), dtype=torch.float32)
        
        # 创建批次
        batch = {
            "ctrl_expressions": ctrl_expressions,
            "pert_emb": pert_emb
        }
        
        # 运行推理
        if return_trajectory:
            predictions = model.get_perturbation_trajectory(batch)
            trajectory = predictions
        else:
            predictions = model(batch)
            trajectory = None
        
        # 转换为 AnnData
        pred_adata = ad.AnnData(
            X=predictions.cpu().numpy(),
            obs=ctrl_data.obs.copy(),
            var=ctrl_data.var.copy()
        )
        
        return pred_adata, trajectory

def analyze_perturbation_effects(
    model: SE_ST_NeuralODE_Model,
    ctrl_data: ad.AnnData,
    pert_emb: torch.Tensor,
    perturbation_name: str
) -> Dict:
    """
    分析扰动效果
    
    Args:
        model: 训练好的模型
        ctrl_data: 控制组数据
        pert_emb: 扰动嵌入
        perturbation_name: 扰动名称
    
    Returns:
        分析结果
    """
    # 准备批次
    batch = {
        "ctrl_expressions": torch.tensor(ctrl_data.X.toarray(), dtype=torch.float32),
        "pert_emb": pert_emb
    }
    
    # 分析扰动动力学
    analysis_results = analyze_perturbation_dynamics(
        model=model,
        batch=batch,
        perturbation_name=perturbation_name
    )
    
    return analysis_results

def main():
    """主推理函数"""
    
    # 设置日志
    logging.basicConfig(level=logging.INFO)
    
    # 加载模型
    model = load_neural_ode_model("checkpoints/best_model.ckpt")
    
    # 加载数据
    ctrl_data = sc.read_h5ad("data/ctrl_data.h5ad")
    pert_emb = torch.load("data/pert_emb.pt")
    
    # 运行推理
    predictions, trajectory = run_neural_ode_inference(
        model=model,
        ctrl_data=ctrl_data,
        pert_emb=pert_emb,
        return_trajectory=True
    )
    
    # 保存结果
    predictions.write_h5ad("results/predictions.h5ad")
    
    if trajectory is not None:
        torch.save(trajectory, "results/trajectory.pt")
    
    # 分析扰动效果
    analysis_results = analyze_perturbation_effects(
        model=model,
        ctrl_data=ctrl_data,
        pert_emb=pert_emb,
        perturbation_name="example_perturbation"
    )
    
    # 可视化结果
    visualize_perturbation_dynamics(analysis_results)
    
    logger.info("Inference completed successfully!")

if __name__ == "__main__":
    main()
