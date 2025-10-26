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
import argparse
from pathlib import Path

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

    # 解析命令行参数
    parser = argparse.ArgumentParser(description="Neural ODE Inference")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint")
    parser.add_argument("--adata", type=str, required=True,
                        help="Path to control AnnData file")
    parser.add_argument("--output", type=str, required=True,
                        help="Path to output predictions")
    parser.add_argument("--perturbation-features", type=str, required=True,
                        help="Path to perturbation features (ESM2 embeddings)")
    parser.add_argument("--se-model-path", type=str, default="SE-600M",
                        help="Path to SE model")
    parser.add_argument("--batch-size", type=int, default=16,
                        help="Batch size for inference")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to use (cuda or cpu)")
    parser.add_argument("--return-trajectory", action="store_true",
                        help="Return full perturbation trajectory")

    args = parser.parse_args()

    # 设置日志
    logging.basicConfig(level=logging.INFO)

    logger.info(f"Loading model from {args.checkpoint}")

    # 加载模型
    model = load_neural_ode_model(args.checkpoint, device=args.device)
    model = model.to(args.device)

    logger.info(f"Loading data from {args.adata}")

    # 加载数据
    ctrl_data = sc.read_h5ad(args.adata)

    logger.info(f"Loading perturbation features from {args.perturbation_features}")

    # 加载扰动嵌入
    pert_features = torch.load(args.perturbation_features, map_location=args.device)

    # 从 AnnData 中获取扰动信息
    if 'perturbation' in ctrl_data.obs.columns:
        perturbations = ctrl_data.obs['perturbation'].unique()
        logger.info(f"Found {len(perturbations)} unique perturbations")
    else:
        logger.warning("No 'perturbation' column found in obs, using first perturbation")
        perturbations = [list(pert_features.keys())[0]]

    # 对每个扰动进行推理
    all_predictions = []

    for pert_name in perturbations:
        logger.info(f"Processing perturbation: {pert_name}")

        # 获取扰动嵌入
        if pert_name in pert_features:
            pert_emb = pert_features[pert_name].unsqueeze(0).to(args.device)
        else:
            logger.warning(f"Perturbation {pert_name} not found in features, skipping")
            continue

        # 运行推理
        predictions, trajectory = run_neural_ode_inference(
            model=model,
            ctrl_data=ctrl_data,
            pert_emb=pert_emb,
            batch_size=args.batch_size,
            return_trajectory=args.return_trajectory
        )

        all_predictions.append(predictions)

        if args.return_trajectory and trajectory is not None:
            trajectory_path = Path(args.output).parent / f"trajectory_{pert_name}.pt"
            torch.save(trajectory, trajectory_path)
            logger.info(f"Saved trajectory to {trajectory_path}")

    # 合并所有预测
    if len(all_predictions) > 0:
        final_predictions = ad.concat(all_predictions)

        logger.info(f"Saving predictions to {args.output}")
        final_predictions.write_h5ad(args.output)

        logger.info("Inference completed successfully!")
    else:
        logger.error("No predictions were generated!")

if __name__ == "__main__":
    main()
