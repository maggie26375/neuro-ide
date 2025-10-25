"""
Integrated Three-Layer Perception Control System.

This module combines:
1. Active Perception Layer: Decides what to observe
2. Temporal Control Layer: Decides when to intervene  
3. Adaptive Representation Layer: Decides how to represent
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
from .active_perception import ActivePerceptionLayer
from .temporal_control import TemporalControlLayer
from .adaptive_representation import AdaptiveRepresentationNetwork

class IntegratedPerceptionSystem(nn.Module):
    """
    集成三层感知控制系统
    
    1. 主动感知层：决定观测什么
    2. 时间控制层：决定何时干预
    3. 自适应表征层：决定如何表征
    """
    
    def __init__(
        self,
        input_dim: int,
        feature_dim: int,
        num_features: int,
        max_dim: int = 512,
        min_dim: int = 64,
        hidden_dim: int = 256,
        num_time_steps: int = 10,
        intervention_dim: int = 128
    ):
        super().__init__()
        
        # 三层系统
        self.active_perception = ActivePerceptionLayer(
            input_dim=input_dim,
            feature_dim=feature_dim,
            num_features=num_features
        )
        
        self.temporal_control = TemporalControlLayer(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_time_steps=num_time_steps,
            intervention_dim=intervention_dim
        )
        
        self.adaptive_representation = AdaptiveRepresentationNetwork(
            input_dim=input_dim,
            max_dim=max_dim,
            min_dim=min_dim
        )
        
        # 协调器 - 需要处理可变维度的 adaptive_repr
        # 使用 max_dim 作为协调器的基准维度
        self.max_dim = max_dim
        self.coordinator = nn.Sequential(
            nn.Linear(input_dim + input_dim + max_dim, 512),
            nn.ReLU(),
            nn.Linear(512, input_dim)
        )

        # 添加投影层将 adaptive_repr 投影到 max_dim
        self.adaptive_repr_projector = nn.Linear(max_dim, max_dim)
    
    def forward(
        self,
        x: torch.Tensor,
        temporal_sequence: torch.Tensor,
        available_features: List[torch.Tensor],
        intervention_history: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """前向传播"""
        
        # 1. 主动感知
        enhanced_x, perception_info = self.active_perception(x, available_features)
        
        # 2. 时间控制
        next_state, control_info = self.temporal_control(
            temporal_sequence, enhanced_x, intervention_history
        )
        
        # 3. 自适应表征
        adaptive_repr, adaptation_info = self.adaptive_representation(next_state)

        # 4. 投影 adaptive_repr 到固定维度
        # 如果 adaptive_repr 维度不是 max_dim，需要先投影
        if adaptive_repr.size(-1) != self.max_dim:
            # 创建一个临时投影层或填充
            if adaptive_repr.size(-1) < self.max_dim:
                # 填充到 max_dim
                padding = torch.zeros(
                    adaptive_repr.size(0),
                    self.max_dim - adaptive_repr.size(-1),
                    device=adaptive_repr.device
                )
                adaptive_repr_fixed = torch.cat([adaptive_repr, padding], dim=-1)
            else:
                # 裁剪到 max_dim
                adaptive_repr_fixed = adaptive_repr[:, :self.max_dim]
        else:
            adaptive_repr_fixed = adaptive_repr

        adaptive_repr_projected = self.adaptive_repr_projector(adaptive_repr_fixed)

        # 5. 协调三层输出
        coordinated_output = self.coordinator(
            torch.cat([enhanced_x, next_state, adaptive_repr_projected], dim=-1)
        )
        
        # 5. 收集所有信息
        system_info = {
            'perception': perception_info,
            'control': control_info,
            'adaptation': adaptation_info,
            'coordinated_output': coordinated_output
        }
        
        return coordinated_output, system_info
    
    def compute_system_loss(
        self,
        output: torch.Tensor,
        target: torch.Tensor,
        system_info: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """计算系统总损失"""
        # 1. 主任务损失
        main_loss = F.mse_loss(output, target)
        
        # 2. 感知损失
        perception_loss = self._compute_perception_loss(system_info['perception'])
        
        # 3. 控制损失
        control_loss = self._compute_control_loss(system_info['control'])
        
        # 4. 适应损失
        adaptation_loss = self.adaptive_representation.compute_adaptation_loss(
            output, target, system_info['adaptation']
        )
        
        # 5. 协调损失
        coordination_loss = F.mse_loss(
            system_info['coordinated_output'], 
            target
        )
        
        # 6. 总损失
        total_loss = (
            main_loss + 
            0.1 * perception_loss + 
            0.1 * control_loss + 
            0.1 * adaptation_loss + 
            0.05 * coordination_loss
        )
        
        return total_loss
    
    def _compute_perception_loss(self, perception_info: Dict[str, torch.Tensor]) -> torch.Tensor:
        """计算感知损失"""
        # 基于不确定性的损失
        uncertainty_loss = perception_info['uncertainty'].mean()
        
        # 基于采样成本的损失
        sampling_cost = perception_info['sampling_cost'].mean()
        
        return uncertainty_loss + 0.1 * sampling_cost
    
    def _compute_control_loss(self, control_info: Dict[str, torch.Tensor]) -> torch.Tensor:
        """计算控制损失"""
        # 基于干预概率的损失
        intervention_loss = control_info['intervention_prob'].mean()
        
        # 基于干预强度的损失
        strength_loss = control_info['intervention_strength'].mean()
        
        return intervention_loss + 0.1 * strength_loss
