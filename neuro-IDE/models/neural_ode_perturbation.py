"""
Neural ODE-based perturbation prediction model for single-cell genomics.

This module implements the core Neural ODE functionality for modeling
continuous cell state dynamics under perturbation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchdiffeq import odeint_adjoint as odeint
from typing import Dict, Optional, Tuple, List
import math

class PerturbationODEFunc(nn.Module):
    """
    扰动动力学函数：dX/dt = f(X, P, t)
    
    核心思想：
    - X: 细胞状态嵌入 (来自 SE Encoder)
    - P: 扰动嵌入 (来自 ESM2)
    - t: 虚拟时间 (扰动作用强度)
    - f: 学习的速度场，描述状态如何演化
    """
    
    def __init__(
        self, 
        state_dim: int, 
        pert_dim: int, 
        hidden_dim: int = 128,
        num_layers: int = 3
    ):
        super().__init__()
        self.state_dim = state_dim
        self.pert_dim = pert_dim
        
        # 构建网络：输入 [X, P, t] -> 输出 dX/dt
        layers = []
        input_dim = state_dim + pert_dim + 1  # +1 for time t
        
        for i in range(num_layers):
            if i == 0:
                layers.append(nn.Linear(input_dim, hidden_dim))
            else:
                layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.Tanh())
        
        layers.append(nn.Linear(hidden_dim, state_dim))
        self.net = nn.Sequential(*layers)
        
        # 可学习的扰动强度调制器
        self.perturbation_modulator = nn.Sequential(
            nn.Linear(pert_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
    
    def forward(self, t: torch.Tensor, x: torch.Tensor, pert_emb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: 时间标量或张量
            x: 当前状态 [batch_size, state_dim]
            pert_emb: 扰动嵌入 [batch_size, pert_dim]

        Returns:
            dx/dt: 状态变化速度 [batch_size, state_dim]
        """
        batch_size = x.shape[0]

        # 处理时间维度
        if t.dim() == 0:  # 标量时间
            t_expanded = t.expand(batch_size, 1)
        else:  # 向量时间
            t_expanded = t.unsqueeze(-1) if t.dim() == 1 else t

        # 确保所有张量都是 2D [batch_size, dim]
        if x.dim() == 3:
            # 如果 x 是 3D，reshape 到 2D
            x = x.reshape(-1, x.shape[-1])
        if pert_emb.dim() == 3:
            # 如果 pert_emb 是 3D，reshape 到 2D
            pert_emb = pert_emb.reshape(-1, pert_emb.shape[-1])

        # 拼接输入：[X, P, t]
        input_features = torch.cat([x, pert_emb, t_expanded], dim=-1)
        
        # 计算基础速度场
        base_velocity = self.net(input_features)
        
        # 扰动强度调制
        perturbation_strength = self.perturbation_modulator(pert_emb)
        modulated_velocity = base_velocity * perturbation_strength
        
        return modulated_velocity


class NeuralODEPerturbationModel(nn.Module):
    """
    基于 Neural ODE 的扰动预测模型
    
    核心流程：
    1. SE Encoder: 原始基因表达 -> 初始状态 x0
    2. Neural ODE: x0 -> x(t) 通过求解 dX/dt = f(X, P, t)
    3. Gene Decoder: x(t) -> 预测基因表达
    """
    
    def __init__(
        self,
        state_dim: int,
        pert_dim: int,
        gene_dim: int,
        ode_hidden_dim: int = 128,
        ode_layers: int = 3,
        time_range: Tuple[float, float] = (0.0, 1.0),
        num_time_points: int = 10
    ):
        super().__init__()
        self.state_dim = state_dim
        self.pert_dim = pert_dim
        self.gene_dim = gene_dim
        
        # Neural ODE 函数
        self.ode_func = PerturbationODEFunc(
            state_dim=state_dim,
            pert_dim=pert_dim,
            hidden_dim=ode_hidden_dim,
            num_layers=ode_layers
        )
        
        # 时间范围
        self.time_range = time_range
        self.time_points = torch.linspace(
            time_range[0], time_range[1], num_time_points
        )
        
        # 基因解码器 (从状态嵌入到基因表达)
        self.gene_decoder = nn.Sequential(
            nn.Linear(state_dim, state_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(state_dim * 2, state_dim),
            nn.ReLU(),
            nn.Linear(state_dim, gene_dim)
        )
        
        # 可学习的时间编码器
        self.time_encoder = nn.Sequential(
            nn.Linear(1, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )
    
    def forward(
        self,
        initial_states: torch.Tensor,
        perturbation_emb: torch.Tensor,
        return_trajectory: bool = False
    ) -> torch.Tensor:
        """
        Args:
            initial_states: 初始状态 [batch_size, state_dim]
            perturbation_emb: 扰动嵌入 [batch_size, pert_dim]
            return_trajectory: 是否返回完整轨迹

        Returns:
            predicted_expressions: 预测基因表达 [batch_size, gene_dim]
            或 trajectory: 完整状态轨迹 [time_points, batch_size, state_dim]
        """
        batch_size = initial_states.shape[0]
        device = initial_states.device

        # 将时间点移到设备上
        time_points = self.time_points.to(device)

        # 创建一个包装类以满足 odeint_adjoint 的要求
        class ODEWrapper(nn.Module):
            def __init__(self, ode_func, pert_emb):
                super().__init__()
                self.ode_func = ode_func
                self.pert_emb = pert_emb

            def forward(self, t, x):
                return self.ode_func(t, x, self.pert_emb)

        ode_wrapper = ODEWrapper(self.ode_func, perturbation_emb)

        # 求解 Neural ODE
        # 使用 odeint_adjoint 自动使用 adjoint 方法进行反向传播
        trajectory = odeint(
            ode_wrapper,
            initial_states,
            time_points,
            rtol=1e-3,
            atol=1e-4,
            method='dopri5'  # 推荐的高精度求解器
        )
        
        if return_trajectory:
            return trajectory
        
        # 取最终时间点的状态
        final_states = trajectory[-1]  # [batch_size, state_dim]
        
        # 解码为基因表达
        predicted_expressions = self.gene_decoder(final_states)
        
        return predicted_expressions
    
    def get_velocity_field(
        self, 
        state_grid: torch.Tensor, 
        perturbation_emb: torch.Tensor,
        time_point: float = 0.5
    ) -> torch.Tensor:
        """
        获取速度场，用于可视化和分析
        
        Args:
            state_grid: 状态网格 [grid_size, state_dim]
            perturbation_emb: 扰动嵌入 [1, pert_dim]
            time_point: 时间点
        
        Returns:
            velocity_field: 速度场 [grid_size, state_dim]
        """
        t = torch.tensor(time_point, device=state_grid.device)
        pert_emb_expanded = perturbation_emb.expand(state_grid.shape[0], -1)
        
        with torch.no_grad():
            velocity = self.ode_func(t, state_grid, pert_emb_expanded)
        
        return velocity