"""
Adaptive Representation Network for dynamic dimensionality adjustment.

This module implements the adaptive representation system that dynamically
adjusts representation dimensions based on task complexity.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple
import math

class AdaptiveRepresentationNetwork(nn.Module):
    """
    自适应表征网络：动态调整表征维度
    
    核心思想：
    1. 基于任务复杂度动态调整表征维度
    2. 使用可学习掩码或动态压缩层
    3. 保持表征质量的同时提高效率
    """
    
    def __init__(
        self,
        input_dim: int,
        max_dim: int = 512,
        min_dim: int = 64,
        num_compression_levels: int = 4,
        compression_ratio: float = 0.8
    ):
        super().__init__()
        self.input_dim = input_dim
        self.max_dim = max_dim
        self.min_dim = min_dim
        self.num_compression_levels = num_compression_levels
        self.compression_ratio = compression_ratio
        
        # 维度决策器
        self.dimension_controller = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        
        # 可学习掩码生成器
        self.mask_generator = nn.Sequential(
            nn.Linear(input_dim, max_dim),
            nn.ReLU(),
            nn.Linear(max_dim, max_dim),
            nn.Sigmoid()
        )
        
        # 动态压缩层
        self.compression_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(max_dim, int(max_dim * (compression_ratio ** i))),
                nn.ReLU(),
                nn.Linear(int(max_dim * (compression_ratio ** i)), max_dim)
            ) for i in range(num_compression_levels)
        ])
        
        # 维度投影器
        self.dimension_projectors = nn.ModuleList([
            nn.Linear(max_dim, dim) for dim in [
                min_dim, 
                int(max_dim * 0.25), 
                int(max_dim * 0.5), 
                int(max_dim * 0.75), 
                max_dim
            ]
        ])
        
        # 表征质量评估器
        self.quality_assessor = nn.Sequential(
            nn.Linear(max_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        
        # 自适应注意力
        self.adaptive_attention = nn.MultiheadAttention(
            embed_dim=max_dim,
            num_heads=8,
            batch_first=True
        )
    
    def forward(
        self, 
        x: torch.Tensor,
        target_complexity: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        前向传播
        
        Args:
            x: 输入特征 [batch_size, input_dim]
            target_complexity: 目标复杂度 [batch_size, 1]
        
        Returns:
            adaptive_repr: 自适应表征
            adaptation_info: 适应信息
        """
        batch_size = x.size(0)
        
        # 1. 决定目标维度
        if target_complexity is not None:
            target_dim_ratio = target_complexity
        else:
            target_dim_ratio = self.dimension_controller(x)
        
        target_dim = int(self.min_dim + (self.max_dim - self.min_dim) * target_dim_ratio)
        
        # 2. 生成可学习掩码
        mask = self.mask_generator(x)  # [batch_size, max_dim]
        
        # 3. 应用动态压缩
        compressed_repr = x
        for i, compression_layer in enumerate(self.compression_layers):
            if i < len(self.compression_layers) - 1:
                compressed_repr = compression_layer(compressed_repr)
        
        # 4. 应用掩码
        masked_repr = compressed_repr * mask
        
        # 5. 自适应注意力
        attended_repr, attention_weights = self.adaptive_attention(
            masked_repr.unsqueeze(1), 
            masked_repr.unsqueeze(1), 
            masked_repr.unsqueeze(1)
        )
        attended_repr = attended_repr.squeeze(1)
        
        # 6. 投影到目标维度
        adaptive_repr = self._project_to_target_dim(attended_repr, target_dim)
        
        # 7. 评估表征质量
        quality_score = self.quality_assessor(adaptive_repr)
        
        # 8. 收集适应信息
        adaptation_info = {
            'target_dim': target_dim,
            'target_dim_ratio': target_dim_ratio,
            'mask': mask,
            'compressed_repr': compressed_repr,
            'attention_weights': attention_weights,
            'quality_score': quality_score,
            'efficiency_ratio': target_dim / self.max_dim
        }
        
        return adaptive_repr, adaptation_info
    
    def _project_to_target_dim(self, x: torch.Tensor, target_dim: int) -> torch.Tensor:
        """投影到目标维度"""
        if target_dim == self.max_dim:
            return x
        elif target_dim == self.min_dim:
            return self.dimension_projectors[0](x)
        else:
            # 选择最接近的投影器
            available_dims = [proj.out_features for proj in self.dimension_projectors]
            closest_idx = min(range(len(available_dims)), 
                            key=lambda i: abs(available_dims[i] - target_dim))
            return self.dimension_projectors[closest_idx](x)
    
    def compute_adaptation_loss(
        self, 
        adaptive_repr: torch.Tensor,
        target: torch.Tensor,
        adaptation_info: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """计算适应损失"""
        # 1. 重构损失
        reconstruction_loss = F.mse_loss(adaptive_repr, target)
        
        # 2. 稀疏性损失（鼓励稀疏表征）
        sparsity_loss = torch.mean(torch.abs(adaptive_repr))
        
        # 3. 质量损失
        quality_loss = 1.0 - adaptation_info['quality_score'].mean()
        
        # 4. 效率损失
        efficiency_loss = 1.0 - adaptation_info['efficiency_ratio']
        
        # 5. 组合损失
        total_loss = (
            reconstruction_loss + 
            0.1 * sparsity_loss + 
            0.2 * quality_loss + 
            0.1 * efficiency_loss
        )
        
        return total_loss
