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
    - X: 细胞状态嵌入 [batch, seq_len, state_dim] (来自 SE Encoder)
    - P: 扰动嵌入 [batch, pert_dim] (来自 ESM2)
    - t: 虚拟时间 (扰动作用强度)
    - f: 学习的速度场，描述状态如何演化

    支持 3D 输入，保留序列结构，可以建模同一 perturbation 下细胞间的交互
    """

    def __init__(
        self,
        state_dim: int,
        pert_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 3,
        use_cell_attention: bool = True,
        num_attention_heads: int = 8
    ):
        super().__init__()
        self.state_dim = state_dim
        self.pert_dim = pert_dim
        self.use_cell_attention = use_cell_attention

        # 细胞间注意力机制 - 让同一 perturbation 下的细胞可以交互
        if use_cell_attention:
            self.cell_attention = nn.MultiheadAttention(
                embed_dim=state_dim,
                num_heads=num_attention_heads,
                dropout=0.1,
                batch_first=True
            )
            self.attention_norm = nn.LayerNorm(state_dim)

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
            x: 当前状态 [batch, seq_len, state_dim] (3D 输入，保留序列结构)
            pert_emb: 扰动嵌入 [batch, pert_dim] (2D 输入)

        Returns:
            dx/dt: 状态变化速度 [batch, seq_len, state_dim] (3D 输出)
        """
        # 支持 3D 输入 [batch, seq_len, state_dim]
        if x.dim() != 3:
            raise ValueError(f"Expected 3D input [batch, seq_len, state_dim], got {x.shape}")

        batch_size, seq_len, state_dim = x.shape

        # 确保 pert_emb 是 2D [batch, pert_dim]
        if pert_emb.dim() == 1:
            pert_emb = pert_emb.unsqueeze(0)  # [pert_dim] -> [1, pert_dim]

        # 如果 pert_emb 维度不对，截取正确的维度
        if pert_emb.shape[-1] != self.pert_dim:
            pert_emb = pert_emb[:, :self.pert_dim]

        # 确保 pert_emb 的 batch 维度匹配
        if pert_emb.shape[0] == 1 and batch_size > 1:
            pert_emb = pert_emb.expand(batch_size, -1)
        elif pert_emb.shape[0] != batch_size:
            raise ValueError(f"Batch size mismatch: x={batch_size}, pert_emb={pert_emb.shape[0]}")

        # 1. 细胞间注意力 - 让同一 perturbation 下的细胞可以交互
        if self.use_cell_attention:
            # Self-attention within each perturbation's cell population
            x_attended, _ = self.cell_attention(x, x, x)
            # Residual connection + layer norm
            x = self.attention_norm(x + x_attended)

        # 2. 扩展 pert_emb 到每个细胞 [batch, pert_dim] -> [batch, seq_len, pert_dim]
        pert_emb_expanded = pert_emb.unsqueeze(1).expand(-1, seq_len, -1)

        # 3. 处理时间维度
        if t.dim() == 0:  # 标量时间
            t_expanded = t.expand(batch_size, seq_len, 1)
        else:
            raise NotImplementedError("Only scalar time is supported in 3D mode")

        # 4. 拼接输入：[X, P, t] -> [batch, seq_len, state_dim + pert_dim + 1]
        input_features = torch.cat([x, pert_emb_expanded, t_expanded], dim=-1)

        # 5. 计算基础速度场
        base_velocity = self.net(input_features)  # [batch, seq_len, state_dim]

        # 6. 扰动强度调制（对每个细胞应用相同的扰动强度）
        perturbation_strength = self.perturbation_modulator(pert_emb)  # [batch, 1]
        perturbation_strength = perturbation_strength.unsqueeze(1)  # [batch, 1, 1]

        # 7. 调制速度场
        modulated_velocity = base_velocity * perturbation_strength  # [batch, seq_len, state_dim]

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
        num_time_points: int = 10,
        use_cell_attention: bool = True,
        num_attention_heads: int = 8
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
            num_layers=ode_layers,
            use_cell_attention=use_cell_attention,
            num_attention_heads=num_attention_heads
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
            initial_states: 初始状态 [batch, seq_len, state_dim] (3D 输入，保留细胞序列结构)
            perturbation_emb: 扰动嵌入 [batch, pert_dim] (2D 输入)
            return_trajectory: 是否返回完整轨迹

        Returns:
            predicted_expressions: 预测基因表达 [batch, seq_len, gene_dim]
            或 trajectory: 完整状态轨迹 [time_points, batch, seq_len, state_dim]
        """
        if initial_states.dim() != 3:
            raise ValueError(f"Expected 3D initial_states [batch, seq_len, state_dim], got {initial_states.shape}")

        batch_size, seq_len, state_dim = initial_states.shape
        device = initial_states.device

        # 确保 perturbation_emb 是 2D
        if perturbation_emb.dim() == 1:
            perturbation_emb = perturbation_emb.unsqueeze(0)

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
        # 输入: [batch, seq_len, state_dim]
        # 输出: [time_points, batch, seq_len, state_dim]
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
        final_states = trajectory[-1]  # [batch, seq_len, state_dim]

        # 解码为基因表达
        # 需要 reshape 来批量处理所有细胞
        final_states_flat = final_states.reshape(-1, state_dim)  # [batch*seq_len, state_dim]
        predicted_expressions_flat = self.gene_decoder(final_states_flat)  # [batch*seq_len, gene_dim]
        predicted_expressions = predicted_expressions_flat.reshape(batch_size, seq_len, self.gene_dim)

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