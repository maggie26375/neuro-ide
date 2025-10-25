"""
Temporal Control Layer for dynamic intervention timing.

This module implements the temporal control system that decides
when to intervene based on time series analysis.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple
import math

class TemporalControlLayer(nn.Module):
    """
    时间控制层：决定何时干预
    
    核心思想：
    1. 基于时间序列预测干预时机
    2. 动态调整干预强度
    3. 平衡干预效果和副作用
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        num_time_steps: int = 10,
        intervention_dim: int = 128,
        patience: int = 3
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_time_steps = num_time_steps
        self.intervention_dim = intervention_dim
        self.patience = patience
        
        # 时间序列编码器
        self.temporal_encoder = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=0.1
        )
        
        # 干预时机预测器
        self.intervention_timing = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
        # 干预强度控制器
        self.intervention_strength = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, intervention_dim),
            nn.Sigmoid()
        )
        
        # 干预类型选择器
        self.intervention_type = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 5),  # 5种干预类型
            nn.Softmax(dim=-1)
        )
        
        # 状态预测器
        self.state_predictor = nn.Sequential(
            nn.Linear(hidden_dim + intervention_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )
        
        # 时间注意力
        self.temporal_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=8,
            batch_first=True
        )
    
    def forward(
        self, 
        temporal_sequence: torch.Tensor,
        current_state: torch.Tensor,
        intervention_history: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        前向传播
        
        Args:
            temporal_sequence: 时间序列 [batch_size, seq_len, input_dim]
            current_state: 当前状态 [batch_size, input_dim]
            intervention_history: 干预历史 [batch_size, seq_len, intervention_dim]
        
        Returns:
            next_state: 下一状态预测
            control_info: 控制信息
        """
        batch_size, seq_len, _ = temporal_sequence.shape
        
        # 1. 时间序列编码
        temporal_output, (hidden, cell) = self.temporal_encoder(temporal_sequence)
        
        # 2. 时间注意力
        attended_temporal, attention_weights = self.temporal_attention(
            temporal_output, temporal_output, temporal_output
        )
        
        # 3. 使用最后一个时间步的隐藏状态
        last_hidden = attended_temporal[:, -1, :]  # [batch_size, hidden_dim]
        
        # 4. 预测干预时机
        intervention_prob = self.intervention_timing(last_hidden)  # [batch_size, 1]
        
        # 5. 预测干预强度
        intervention_strength = self.intervention_strength(last_hidden)  # [batch_size, intervention_dim]
        
        # 6. 选择干预类型
        intervention_type = self.intervention_type(last_hidden)  # [batch_size, 5]
        
        # 7. 决定是否干预
        should_intervene = self._should_intervene(
            intervention_prob, 
            intervention_history,
            current_state
        )
        
        # 8. 计算干预效果
        if should_intervene.any():
            intervention_effect = self._compute_intervention_effect(
                intervention_strength,
                intervention_type,
                current_state
            )
        else:
            intervention_effect = torch.zeros_like(current_state)
        
        # 9. 预测下一状态
        next_state = self._predict_next_state(
            current_state,
            intervention_effect,
            last_hidden
        )
        
        # 10. 收集控制信息
        control_info = {
            'intervention_prob': intervention_prob,
            'intervention_strength': intervention_strength,
            'intervention_type': intervention_type,
            'should_intervene': should_intervene,
            'intervention_effect': intervention_effect,
            'attention_weights': attention_weights,
            'temporal_hidden': last_hidden
        }
        
        return next_state, control_info
    
    def _should_intervene(
        self, 
        intervention_prob: torch.Tensor,
        intervention_history: Optional[torch.Tensor],
        current_state: torch.Tensor
    ) -> torch.Tensor:
        """决定是否干预"""
        # 基于概率阈值
        prob_threshold = 0.5
        should_intervene = intervention_prob > prob_threshold
        
        # 基于干预历史（避免过度干预）
        if intervention_history is not None:
            recent_interventions = intervention_history[:, -self.patience:].sum(dim=1)
            should_intervene = should_intervene & (recent_interventions < self.patience)
        
        return should_intervene
    
    def _compute_intervention_effect(
        self,
        intervention_strength: torch.Tensor,
        intervention_type: torch.Tensor,
        current_state: torch.Tensor
    ) -> torch.Tensor:
        """计算干预效果"""
        # 基于干预类型和强度计算效果
        intervention_effect = torch.zeros_like(current_state)
        
        for i in range(intervention_strength.size(1)):
            # 每种干预类型的效果
            type_effect = intervention_type[:, i:i+1] * intervention_strength[:, i:i+1]
            intervention_effect += type_effect * current_state
        
        return intervention_effect
    
    def _predict_next_state(
        self,
        current_state: torch.Tensor,
        intervention_effect: torch.Tensor,
        temporal_hidden: torch.Tensor
    ) -> torch.Tensor:
        """预测下一状态"""
        # 组合当前状态、干预效果和时间信息
        combined_input = torch.cat([
            current_state,
            intervention_effect,
            temporal_hidden
        ], dim=-1)
        
        # 预测下一状态
        next_state = self.state_predictor(combined_input)
        
        return next_state
