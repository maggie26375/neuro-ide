"""
Active Perception Layer for intelligent feature selection and sampling.

This module implements the active perception system that decides
what features to observe and sample based on current state.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
import math

class ActivePerceptionLayer(nn.Module):
    """
    主动感知层：决定是否观测/采样特征
    
    核心思想：
    1. 基于当前状态决定是否需要更多信息
    2. 动态选择观测特征
    3. 平衡信息获取成本和预测准确性
    """
    
    def __init__(
        self,
        input_dim: int,
        feature_dim: int,
        num_features: int,
        attention_dim: int = 128,
        uncertainty_threshold: float = 0.5,
        sampling_budget: int = 10
    ):
        super().__init__()
        self.input_dim = input_dim
        self.feature_dim = feature_dim
        self.num_features = num_features
        self.attention_dim = attention_dim
        self.uncertainty_threshold = uncertainty_threshold
        self.sampling_budget = sampling_budget
        
        # 不确定性估计器
        self.uncertainty_estimator = nn.Sequential(
            nn.Linear(input_dim, attention_dim),
            nn.ReLU(),
            nn.Linear(attention_dim, 1),
            nn.Sigmoid()
        )
        
        # 特征重要性评估器
        self.feature_importance = nn.Sequential(
            nn.Linear(input_dim, attention_dim),
            nn.ReLU(),
            nn.Linear(attention_dim, num_features),
            nn.Softmax(dim=-1)
        )
        
        # 采样决策器
        self.sampling_decision = nn.Sequential(
            nn.Linear(input_dim + num_features, attention_dim),
            nn.ReLU(),
            nn.Linear(attention_dim, num_features),
            nn.Sigmoid()
        )
        
        # 特征编码器
        self.feature_encoder = nn.ModuleList([
            nn.Linear(feature_dim, attention_dim) 
            for _ in range(num_features)
        ])
        
        # 注意力机制
        self.attention = nn.MultiheadAttention(
            embed_dim=attention_dim,
            num_heads=8,
            batch_first=True
        )
        
        # 输出投影
        self.output_projection = nn.Linear(attention_dim, input_dim)
    
    def forward(
        self, 
        x: torch.Tensor, 
        available_features: List[torch.Tensor],
        sampling_costs: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        前向传播
        
        Args:
            x: 输入状态 [batch_size, input_dim]
            available_features: 可用特征列表 [num_features, batch_size, feature_dim]
            sampling_costs: 采样成本 [num_features]
        
        Returns:
            enhanced_x: 增强后的状态
            perception_info: 感知信息
        """
        batch_size = x.size(0)
        
        # 1. 估计不确定性
        uncertainty = self.uncertainty_estimator(x)  # [batch_size, 1]
        
        # 2. 评估特征重要性
        feature_importance = self.feature_importance(x)  # [batch_size, num_features]
        
        # 3. 采样决策
        sampling_input = torch.cat([x, feature_importance], dim=-1)
        sampling_probs = self.sampling_decision(sampling_input)  # [batch_size, num_features]
        
        # 4. 基于不确定性和成本调整采样概率
        if sampling_costs is not None:
            cost_adjusted_probs = sampling_probs / (1 + sampling_costs.unsqueeze(0))
        else:
            cost_adjusted_probs = sampling_probs
        
        # 5. 选择要采样的特征
        selected_features = self._select_features(
            cost_adjusted_probs, 
            uncertainty, 
            self.sampling_budget
        )
        
        # 6. 编码选中的特征
        encoded_features = self._encode_selected_features(
            selected_features, 
            available_features
        )
        
        # 7. 注意力融合
        if encoded_features.size(1) > 0:
            attended_features, attention_weights = self.attention(
                encoded_features, encoded_features, encoded_features
            )
            attended_features = attended_features.mean(dim=1)  # [batch_size, attention_dim]
        else:
            attended_features = torch.zeros(batch_size, self.attention_dim, device=x.device)
        
        # 8. 输出投影
        enhanced_x = self.output_projection(attended_features)
        
        # 9. 残差连接
        enhanced_x = x + enhanced_x
        
        # 10. 收集感知信息
        perception_info = {
            'uncertainty': uncertainty,
            'feature_importance': feature_importance,
            'sampling_probs': sampling_probs,
            'selected_features': selected_features,
            'attention_weights': attention_weights if encoded_features.size(1) > 0 else None,
            'sampling_cost': self._compute_sampling_cost(selected_features, sampling_costs)
        }
        
        return enhanced_x, perception_info
    
    def _select_features(
        self, 
        sampling_probs: torch.Tensor, 
        uncertainty: torch.Tensor,
        budget: int
    ) -> torch.Tensor:
        """选择要采样的特征"""
        batch_size, num_features = sampling_probs.shape
        
        # 基于不确定性调整采样概率
        adjusted_probs = sampling_probs * uncertainty
        
        # 选择前k个特征
        _, top_indices = torch.topk(adjusted_probs, min(budget, num_features), dim=-1)
        
        # 创建选择掩码
        selected_mask = torch.zeros_like(sampling_probs)
        selected_mask.scatter_(1, top_indices, 1.0)
        
        return selected_mask
    
    def _encode_selected_features(
        self, 
        selected_mask: torch.Tensor, 
        available_features: List[torch.Tensor]
    ) -> torch.Tensor:
        """编码选中的特征"""
        batch_size = selected_mask.size(0)
        encoded_features = []
        
        for i, feature in enumerate(available_features):
            if selected_mask[:, i].sum() > 0:  # 如果这个特征被选中
                # 编码特征
                encoded = self.feature_encoder[i](feature)
                # 应用选择掩码
                masked_encoded = encoded * selected_mask[:, i:i+1].unsqueeze(-1)
                encoded_features.append(masked_encoded)
        
        if encoded_features:
            return torch.stack(encoded_features, dim=1)  # [batch_size, num_selected, attention_dim]
        else:
            return torch.empty(batch_size, 0, self.attention_dim, device=selected_mask.device)
    
    def _compute_sampling_cost(
        self, 
        selected_mask: torch.Tensor, 
        sampling_costs: Optional[torch.Tensor]
    ) -> torch.Tensor:
        """计算采样成本"""
        if sampling_costs is None:
            return selected_mask.sum(dim=-1).float()
        else:
            return (selected_mask * sampling_costs.unsqueeze(0)).sum(dim=-1)