"""
Analysis and visualization tools for Neural ODE models.

This module provides utilities for analyzing perturbation dynamics,
visualizing trajectories, and evaluating model performance.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Tuple, List, Optional
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

def analyze_perturbation_dynamics(
    model: torch.nn.Module,
    batch: Dict[str, torch.Tensor],
    perturbation_name: str,
    num_time_points: int = 20
) -> Dict[str, np.ndarray]:
    """
    分析扰动动力学
    
    Args:
        model: 训练好的模型
        batch: 输入批次
        perturbation_name: 扰动名称
        num_time_points: 时间点数
    
    Returns:
        analysis_results: 包含轨迹、速度场等分析结果
    """
    with torch.no_grad():
        # 获取扰动轨迹
        trajectory = model.get_perturbation_trajectory(batch, num_time_points)
        
        # 计算速度场
        initial_states = model.encode_cells_to_state(batch["ctrl_expressions"])
        pert_emb = batch["pert_emb"]
        
        # 在状态空间中创建网格
        state_min = initial_states.min(dim=0)[0]
        state_max = initial_states.max(dim=0)[0]
        
        # 选择前两个维度进行可视化
        grid_size = 20
        x_grid = torch.linspace(state_min[0], state_max[0], grid_size)
        y_grid = torch.linspace(state_min[1], state_max[1], grid_size)
        X, Y = torch.meshgrid(x_grid, y_grid, indexing='ij')
        
        # 创建状态网格
        state_grid = torch.stack([X.flatten(), Y.flatten()], dim=1)
        state_grid = torch.cat([
            state_grid, 
            torch.zeros(state_grid.shape[0], model.st_hidden_dim - 2)
        ], dim=1)
        
        # 计算速度场
        velocity_field = model.get_velocity_field(
            batch, state_grid
        )
        
        return {
            'trajectory': trajectory.cpu().numpy(),
            'velocity_field': velocity_field.cpu().numpy(),
            'state_grid': state_grid.cpu().numpy(),
            'perturbation_name': perturbation_name
        }

def visualize_perturbation_dynamics(
    analysis_results: Dict[str, np.ndarray],
    save_path: Optional[str] = None
) -> None:
    """
    可视化扰动动力学
    
    Args:
        analysis_results: 分析结果
        save_path: 保存路径
    """
    trajectory = analysis_results['trajectory']
    velocity_field = analysis_results['velocity_field']
    state_grid = analysis_results['state_grid']
    perturbation_name = analysis_results['perturbation_name']
    
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
    
    # 轨迹可视化
    ax1.plot(trajectory[:, 0, 0], trajectory[:, 0, 1], 'b-', linewidth=2, label='Trajectory')
    ax1.scatter(trajectory[0, 0, 0], trajectory[0, 0, 1], color='green', s=100, label='Start')
    ax1.scatter(trajectory[-1, 0, 0], trajectory[-1, 0, 1], color='red', s=100, label='End')
    ax1.set_xlabel('State Dim 1')
    ax1.set_ylabel('State Dim 2')
    ax1.set_title(f'Perturbation Trajectory: {perturbation_name}')
    ax1.legend()
    ax1.grid(True)
    
    # 速度场可视化
    X = state_grid[:, 0].reshape(20, 20)
    Y = state_grid[:, 1].reshape(20, 20)
    U = velocity_field[:, 0].reshape(20, 20)
    V = velocity_field[:, 1].reshape(20, 20)
    
    ax2.quiver(X, Y, U, V, alpha=0.7, scale=50)
    ax2.set_xlabel('State Dim 1')
    ax2.set_ylabel('State Dim 2')
    ax2.set_title('Velocity Field')
    ax2.grid(True)
    
    # 状态演化可视化
    time_points = np.linspace(0, 1, trajectory.shape[0])
    ax3.plot(time_points, trajectory[:, 0, 0], 'b-', label='Dim 1')
    ax3.plot(time_points, trajectory[:, 0, 1], 'r-', label='Dim 2')
    ax3.set_xlabel('Time')
    ax3.set_ylabel('State Value')
    ax3.set_title('State Evolution Over Time')
    ax3.legend()
    ax3.grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()

def plot_perturbation_comparison(
    models: List[torch.nn.Module],
    batch: Dict[str, torch.Tensor],
    model_names: List[str],
    save_path: Optional[str] = None
) -> None:
    """
    比较不同模型的扰动预测
    
    Args:
        models: 模型列表
        batch: 输入批次
        model_names: 模型名称列表
        save_path: 保存路径
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()
    
    for i, (model, name) in enumerate(zip(models, model_names)):
        if i >= 4:
            break
            
        with torch.no_grad():
            # 获取预测
            predictions = model(batch)
            
            # 获取轨迹（如果是 Neural ODE 模型）
            if hasattr(model, 'get_perturbation_trajectory'):
                trajectory = model.get_perturbation_trajectory(batch)
                axes[i].plot(trajectory[:, 0, 0], trajectory[:, 0, 1], 'b-', linewidth=2)
                axes[i].set_title(f'{name} - Trajectory')
            else:
                # 对于非 Neural ODE 模型，显示预测分布
                pred_np = predictions.cpu().numpy()
                axes[i].hist(pred_np.flatten(), bins=50, alpha=0.7)
                axes[i].set_title(f'{name} - Prediction Distribution')
            
            axes[i].grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()

def compute_perturbation_metrics(
    model: torch.nn.Module,
    test_dataloader: torch.utils.data.DataLoader,
    device: torch.device
) -> Dict[str, float]:
    """
    计算扰动预测指标
    
    Args:
        model: 训练好的模型
        test_dataloader: 测试数据加载器
        device: 设备
    
    Returns:
        metrics: 评估指标
    """
    model.eval()
    total_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for batch in test_dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            
            # 获取预测
            predictions = model(batch)
            targets = batch.get("target", batch["ctrl_expressions"])
            
            # 计算损失
            loss = torch.nn.functional.mse_loss(predictions, targets)
            total_loss += loss.item()
            num_batches += 1
    
    avg_loss = total_loss / num_batches
    
    return {
        'mse_loss': avg_loss,
        'rmse_loss': np.sqrt(avg_loss),
        'r2_score': 1.0 - avg_loss / np.var(targets.cpu().numpy())
    }

def visualize_embedding_space(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    method: str = 'tsne',
    save_path: Optional[str] = None
) -> None:
    """
    可视化嵌入空间
    
    Args:
        model: 训练好的模型
        dataloader: 数据加载器
        device: 设备
        method: 降维方法 ('tsne' 或 'pca')
        save_path: 保存路径
    """
    model.eval()
    embeddings = []
    labels = []
    
    with torch.no_grad():
        for batch in dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            
            # 获取嵌入
            if hasattr(model, 'encode_cells_to_state'):
                embedding = model.encode_cells_to_state(batch["ctrl_expressions"])
            else:
                embedding = model(batch)
            
            embeddings.append(embedding.cpu().numpy())
            
            # 获取标签（如果有的话）
            if "perturbation" in batch:
                labels.extend(batch["perturbation"])
    
    # 合并嵌入
    embeddings = np.concatenate(embeddings, axis=0)
    
    # 降维
    if method == 'tsne':
        reducer = TSNE(n_components=2, random_state=42)
    else:
        reducer = PCA(n_components=2)
    
    embeddings_2d = reducer.fit_transform(embeddings)
    
    # 可视化
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], 
                         c=range(len(embeddings_2d)), cmap='viridis', alpha=0.7)
    plt.colorbar(scatter)
    plt.title(f'Embedding Space Visualization ({method.upper()})')
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()
