"""
Example script for training and using Neural ODE-based perturbation prediction.

This script demonstrates how to use the Neural ODE integration for
single-cell perturbation prediction.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List

from se_st_combined.models.se_st_combined_neural_ode import SE_ST_NeuralODE_Model
from se_st_combined.utils.neural_ode_analysis import (
    analyze_perturbation_dynamics,
    visualize_perturbation_dynamics,
    compute_perturbation_metrics
)

def create_sample_data(batch_size: int = 16, num_genes: int = 18080) -> Dict[str, torch.Tensor]:
    """创建示例数据"""
    # 创建随机基因表达数据
    ctrl_expressions = torch.randn(batch_size * 128, num_genes)  # 细胞句子
    
    # 创建扰动嵌入
    pert_emb = torch.randn(batch_size, 1280)  # ESM2 特征
    
    return {
        "ctrl_expressions": ctrl_expressions,
        "pert_emb": pert_emb
    }

def train_neural_ode_example():
    """训练 Neural ODE 模型示例"""
    
    # 创建模型
    model = SE_ST_NeuralODE_Model(
        input_dim=18080,
        pert_dim=1280,
        st_hidden_dim=512,
        use_neural_ode=True,
        ode_hidden_dim=128,
        ode_layers=3,
        time_range=(0.0, 1.0),
        num_time_points=10
    )
    
    print("Model created successfully!")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # 创建示例数据
    batch = create_sample_data()
    
    # 前向传播
    with torch.no_grad():
        predictions = model(batch)
        print(f"Predictions shape: {predictions.shape}")
        
        # 获取扰动轨迹
        trajectory = model.get_perturbation_trajectory(batch)
        print(f"Trajectory shape: {trajectory.shape}")
    
    return model, batch

def analyze_perturbation_example(model, batch):
    """分析扰动动力学示例"""
    
    # 分析扰动动力学
    analysis_results = analyze_perturbation_dynamics(
        model=model,
        batch=batch,
        perturbation_name="example_perturbation"
    )
    
    # 可视化结果
    visualize_perturbation_dynamics(analysis_results)
    
    return analysis_results

def compare_models_example():
    """比较不同模型示例"""
    
    # 创建 Neural ODE 模型
    neural_ode_model = SE_ST_NeuralODE_Model(
        input_dim=18080,
        pert_dim=1280,
        st_hidden_dim=512,
        use_neural_ode=True
    )
    
    # 创建传统模型
    traditional_model = SE_ST_NeuralODE_Model(
        input_dim=18080,
        pert_dim=1280,
        st_hidden_dim=512,
        use_neural_ode=False
    )
    
    # 创建示例数据
    batch = create_sample_data()
    
    # 比较预测
    with torch.no_grad():
        neural_ode_pred = neural_ode_model(batch)
        traditional_pred = traditional_model(batch)
        
        print(f"Neural ODE prediction shape: {neural_ode_pred.shape}")
        print(f"Traditional prediction shape: {traditional_pred.shape}")
        
        # 计算差异
        diff = torch.abs(neural_ode_pred - traditional_pred).mean()
        print(f"Mean absolute difference: {diff.item():.4f}")

def main():
    """主函数"""
    print("🚀 Neural ODE Perturbation Prediction Example")
    print("=" * 50)
    
    # 1. 训练模型
    print("\n1. Creating and training model...")
    model, batch = train_neural_ode_example()
    
    # 2. 分析扰动动力学
    print("\n2. Analyzing perturbation dynamics...")
    analysis_results = analyze_perturbation_example(model, batch)
    
    # 3. 比较模型
    print("\n3. Comparing different models...")
    compare_models_example()
    
    # 4. 计算指标
    print("\n4. Computing performance metrics...")
    metrics = compute_perturbation_metrics(
        model=model,
        test_dataloader=None,  # 在实际使用中需要真实的数据加载器
        device=torch.device('cpu')
    )
    print(f"Metrics: {metrics}")
    
    print("\n✅ Example completed successfully!")

if __name__ == "__main__":
    main()
