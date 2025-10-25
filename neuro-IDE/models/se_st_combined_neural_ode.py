"""
SE+ST Combined Model with Neural ODE Integration.

This module integrates Neural ODE functionality into the existing
SE+ST Combined model architecture.
"""

import torch
import torch.nn as nn
from typing import Dict, Tuple, Optional
from .se_st_combined import SE_ST_CombinedModel
from .neural_ode_perturbation import NeuralODEPerturbationModel

class SE_ST_NeuralODE_Model(SE_ST_CombinedModel):
    """
    集成 Neural ODE 的 SE+ST 模型
    
    架构：
    SE Encoder -> Neural ODE -> Gene Decoder
    """
    
    def __init__(
        self,
        use_neural_ode: bool = True,
        ode_hidden_dim: int = 128,
        ode_layers: int = 3,
        time_range: Tuple[float, float] = (0.0, 1.0),
        num_time_points: int = 10,
        use_cell_attention: bool = True,
        num_attention_heads: int = 8,
        **kwargs
    ):
        # 先调用父类初始化（会设置 self.state_dim）
        super().__init__(**kwargs)

        self.use_neural_ode = use_neural_ode

        if self.use_neural_ode:
            # 父类初始化后，self.state_dim 已经被设置
            # SE 模型输出的是 state_dim，我们需要将其映射到 st_hidden_dim
            # 如果 SE 输出维度与 ST 隐藏维度不同，添加映射层
            if self.state_dim != self.st_hidden_dim:
                self.state_projection = nn.Linear(self.state_dim, self.st_hidden_dim)
            else:
                self.state_projection = nn.Identity()

            # 替换 ST 模型为 Neural ODE 模型
            self.neural_ode_model = NeuralODEPerturbationModel(
                state_dim=self.st_hidden_dim,
                pert_dim=self.pert_dim,
                gene_dim=self.input_dim,  # 基因维度
                ode_hidden_dim=ode_hidden_dim,
                ode_layers=ode_layers,
                time_range=time_range,
                num_time_points=num_time_points,
                use_cell_attention=use_cell_attention,
                num_attention_heads=num_attention_heads
            )

            # 禁用原有 ST 模型
            self.st_model = None
    
    def forward(self, batch: Dict[str, torch.Tensor], padded: bool = True) -> torch.Tensor:
        # 支持两种键名：ctrl_expressions 和 ctrl_cell_emb
        ctrl_expressions = batch.get("ctrl_expressions", batch.get("ctrl_cell_emb"))  # [B*S, N_genes]
        pert_emb = batch["pert_emb"]                 # [B, pert_dim]

        # 1. SE Encoder: 基因表达 -> 状态嵌入
        initial_states_flat = self.encode_cells_to_state(ctrl_expressions)  # [B*S, se_output_dim]

        # 如果使用 Neural ODE，映射到 ST 隐藏维度
        if self.use_neural_ode:
            initial_states_flat = self.state_projection(initial_states_flat)  # [B*S, st_hidden_dim]

        # 处理 pert_emb 维度
        # 检查 pert_emb 是否已经被扩展成 [B*S, pert_dim]
        if pert_emb.shape[0] == initial_states_flat.shape[0]:
            # pert_emb 已经被扩展了，需要恢复到 [B, pert_dim]
            # 使用 st_cell_set_len 来确定真实的 batch_size
            if padded:
                # 检查是否样本数小于 st_cell_set_len（验证集可能较小）
                if initial_states_flat.shape[0] < self.st_cell_set_len:
                    # 样本数小于 st_cell_set_len，说明这是一个小批次
                    # 此时所有细胞可能共享同一个 perturbation
                    batch_size = 1
                    pert_emb = pert_emb[0:1]
                else:
                    batch_size = initial_states_flat.shape[0] // self.st_cell_set_len
                    # 每隔 st_cell_set_len 取一个 perturbation（它们应该是重复的）
                    pert_emb = pert_emb[::self.st_cell_set_len]  # [B*S, pert_dim] -> [B, pert_dim]
            else:
                # 如果不是 padded，假设所有细胞共享同一个 perturbation
                batch_size = 1
                pert_emb = pert_emb[0:1]
        elif pert_emb.dim() == 1:
            pert_emb = pert_emb.unsqueeze(0)  # [pert_dim] -> [1, pert_dim]
            batch_size = 1
        else:
            # pert_emb 是正确的 [B, pert_dim] 形状
            batch_size = pert_emb.shape[0]

        cell_sentence_len = initial_states_flat.shape[0] // batch_size

        if self.use_neural_ode:
            # 2. Reshape 到 3D [batch, seq_len, state_dim]
            # Neural ODE 现在支持 3D 输入，保留序列结构
            initial_states_3d = initial_states_flat.reshape(batch_size, cell_sentence_len, self.st_hidden_dim)

            # pert_emb 保持 2D [batch, pert_dim]
            # Neural ODE 会在内部扩展到每个细胞

            # 3. Neural ODE: 学习细胞群体的状态演化（支持细胞间交互）
            predictions = self.neural_ode_model(
                initial_states_3d,      # [batch, seq_len, state_dim]
                pert_emb                # [batch, pert_dim]
            )  # 返回 [batch, seq_len, gene_dim]

            # 4. Flatten 回 2D 以匹配后续处理
            predictions = predictions.reshape(-1, self.output_dim)  # [B*S, gene_dim]
        else:
            # 使用原有 ST 模型
            pert_emb_expanded = pert_emb.unsqueeze(1).repeat(1, cell_sentence_len, 1).reshape(-1, self.pert_dim)
            predictions = self.st_model(initial_states_flat, pert_emb_expanded)

        return predictions
    
    def get_perturbation_trajectory(
        self,
        batch: Dict[str, torch.Tensor],
        num_time_points: int = 20
    ) -> torch.Tensor:
        """
        获取扰动轨迹，用于分析细胞状态演化

        Returns:
            trajectory: [num_time_points, batch, seq_len, state_dim]
        """
        if not self.use_neural_ode:
            raise ValueError("Neural ODE not enabled")

        # 支持两种键名：ctrl_expressions 和 ctrl_cell_emb
        ctrl_expressions = batch.get("ctrl_expressions", batch.get("ctrl_cell_emb"))
        pert_emb = batch["pert_emb"]

        # SE Encoder
        initial_states_flat = self.encode_cells_to_state(ctrl_expressions)
        initial_states_flat = self.state_projection(initial_states_flat)

        # 处理 pert_emb 维度（与 forward 方法相同的逻辑）
        if pert_emb.shape[0] == initial_states_flat.shape[0]:
            # pert_emb 已经被扩展了，需要恢复到 [B, pert_dim]
            if initial_states_flat.shape[0] < self.st_cell_set_len:
                batch_size = 1
                pert_emb = pert_emb[0:1]
            else:
                batch_size = initial_states_flat.shape[0] // self.st_cell_set_len
                pert_emb = pert_emb[::self.st_cell_set_len]
        elif pert_emb.dim() == 1:
            pert_emb = pert_emb.unsqueeze(0)
            batch_size = 1
        else:
            batch_size = pert_emb.shape[0]

        cell_sentence_len = initial_states_flat.shape[0] // batch_size

        # Reshape 到 3D
        initial_states_3d = initial_states_flat.reshape(batch_size, cell_sentence_len, self.st_hidden_dim)

        # 获取完整轨迹
        trajectory = self.neural_ode_model(
            initial_states_3d,
            pert_emb,
            return_trajectory=True
        )

        return trajectory
    
    def get_velocity_field(
        self,
        batch: Dict[str, torch.Tensor],
        state_grid: torch.Tensor,
        time_point: float = 0.5
    ) -> torch.Tensor:
        """
        获取速度场，用于可视化和分析
        
        Args:
            batch: 输入批次
            state_grid: 状态网格 [grid_size, state_dim]
            time_point: 时间点
        
        Returns:
            velocity_field: 速度场 [grid_size, state_dim]
        """
        if not self.use_neural_ode:
            raise ValueError("Neural ODE not enabled")
        
        pert_emb = batch["pert_emb"]
        
        # 处理扰动嵌入
        pert_emb_expanded = pert_emb[:1]  # 取第一个样本
        
        # 获取速度场
        velocity_field = self.neural_ode_model.get_velocity_field(
            state_grid, pert_emb_expanded, time_point
        )
        
        return velocity_field
    
    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int, padded: bool = True) -> torch.Tensor:
        """训练步骤，支持 Neural ODE"""
        predictions = self.forward(batch, padded=padded)
        target = batch["pert_cell_emb"]
        
        if padded:
            predictions = predictions.reshape(-1, self.st_cell_set_len, self.output_dim)
            target = target.reshape(-1, self.st_cell_set_len, self.output_dim)
        else:
            predictions = predictions.reshape(1, -1, self.output_dim)
            target = target.reshape(1, -1, self.output_dim)
        
        if self.use_neural_ode:
            # 使用 Neural ODE 的损失函数
            loss = torch.nn.functional.mse_loss(predictions, target)
        else:
            # 使用原有 ST 模型的损失函数
            loss = self.st_model.loss_fn(predictions, target).nanmean()
        
        self.log("train_loss", loss)
        return loss
    
    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int, padded: bool = True) -> torch.Tensor:
        """验证步骤，支持 Neural ODE"""
        predictions = self.forward(batch, padded=padded)
        target = batch["pert_cell_emb"]

        if padded:
            predictions = predictions.reshape(-1, self.st_cell_set_len, self.output_dim)
            target = target.reshape(-1, self.st_cell_set_len, self.output_dim)
        else:
            predictions = predictions.reshape(1, -1, self.output_dim)
            target = target.reshape(1, -1, self.output_dim)

        if self.use_neural_ode:
            # 使用 Neural ODE 的损失函数
            loss = torch.nn.functional.mse_loss(predictions, target)
        else:
            # 使用原有 ST 模型的损失函数
            loss = self.st_model.loss_fn(predictions, target).nanmean()

        self.log("val_loss", loss)
        return loss

    def configure_optimizers(self):
        """配置优化器，支持 Neural ODE"""
        if self.use_neural_ode:
            # 如果使用 Neural ODE
            if self.freeze_se_model:
                # 只优化 Neural ODE 模型参数
                optimizer = torch.optim.Adam(self.neural_ode_model.parameters(), lr=self.lr)
            else:
                # 优化整个模型（包括 SE 模型）
                optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        else:
            # 如果不使用 Neural ODE，调用父类方法
            optimizer = super().configure_optimizers()

        return optimizer