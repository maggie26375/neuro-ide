# Neural ODE 3D Architecture - 细胞群体动力学建模

## 重大改进说明

### 问题背景

之前的 Neural ODE 实现将所有细胞都 flatten 成 2D 张量 `[batch*seq_len, state_dim]`，这样做存在以下问题：

1. **丢失序列结构信息**：模型不知道哪些细胞属于同一个 perturbation 实验
2. **无法建模细胞间交互**：同一 perturbation 下的细胞应该有某种关联（如细胞间通讯、群体效应）
3. **不符合生物学意义**：细胞群体的演化应该是一个整体过程，而非独立演化

### 新架构设计

现在的 Neural ODE 完全支持 **3D 输入** `[batch, seq_len, state_dim]`，保留细胞序列结构，并通过注意力机制建模细胞间交互。

#### 核心改进

1. **3D 张量流**
   ```
   输入：[batch, seq_len, state_dim]
   ↓
   细胞间注意力 (Self-Attention)
   ↓
   ODE 求解：dX/dt = f(X, P, t)
   ↓
   输出：[batch, seq_len, gene_dim]
   ```

2. **细胞间注意力机制**
   ```python
   # 在 PerturbationODEFunc 中
   self.cell_attention = nn.MultiheadAttention(
       embed_dim=state_dim,
       num_heads=8,
       batch_first=True
   )
   ```

   这让同一个 perturbation 下的细胞可以相互感知和影响，建模细胞群体的集体行为。

3. **扰动强度调制**
   ```python
   # 对整个细胞群体应用统一的扰动强度
   perturbation_strength = self.perturbation_modulator(pert_emb)  # [batch, 1]
   modulated_velocity = base_velocity * perturbation_strength
   ```

## 架构对比

### 旧架构（2D）
```
初始状态：[B*S, state_dim]  # 丢失了哪些细胞属于同一个实验
    ↓
ODE 求解（每个细胞独立演化）
    ↓
输出：[B*S, gene_dim]
```

**问题**：
- ✗ 128 个细胞完全独立，没有交互
- ✗ 丢失了序列结构信息
- ✗ 无法建模群体效应

### 新架构（3D）
```
初始状态：[B, S, state_dim]  # 保留了细胞序列结构
    ↓
细胞间注意力（同一 perturbation 下的细胞可以交互）
    ↓
ODE 求解（群体演化）
    ↓
输出：[B, S, gene_dim]
```

**优势**：
- ✓ 保留序列结构
- ✓ 支持细胞间交互
- ✓ 符合生物学意义
- ✓ 更强的表达能力

## 技术细节

### 1. PerturbationODEFunc (neural_ode_perturbation.py)

**新参数**：
- `use_cell_attention: bool = True` - 是否使用细胞间注意力
- `num_attention_heads: int = 8` - 注意力头数

**核心逻辑**：
```python
def forward(self, t, x, pert_emb):
    # x: [batch, seq_len, state_dim]
    # pert_emb: [batch, pert_dim]

    # 1. 细胞间注意力
    x_attended, _ = self.cell_attention(x, x, x)
    x = self.attention_norm(x + x_attended)

    # 2. 扩展 pert_emb 到每个细胞
    pert_emb_expanded = pert_emb.unsqueeze(1).expand(-1, seq_len, -1)

    # 3. 拼接 [X, P, t]
    input_features = torch.cat([x, pert_emb_expanded, t_expanded], dim=-1)

    # 4. 计算速度场
    base_velocity = self.net(input_features)

    # 5. 扰动强度调制
    perturbation_strength = self.perturbation_modulator(pert_emb).unsqueeze(1)
    modulated_velocity = base_velocity * perturbation_strength

    return modulated_velocity  # [batch, seq_len, state_dim]
```

### 2. NeuralODEPerturbationModel

**输入/输出维度**：
- 输入：`initial_states: [batch, seq_len, state_dim]`
- 输出：`predictions: [batch, seq_len, gene_dim]`
- 轨迹：`trajectory: [time_points, batch, seq_len, state_dim]`

**ODE 求解**：
```python
trajectory = odeint(
    ode_wrapper,
    initial_states,  # [batch, seq_len, state_dim]
    time_points,
    method='dopri5'
)
```

### 3. SE_ST_NeuralODE_Model

**数据流**：
```python
# 1. SE Encoder
ctrl_expressions: [B*S, N_genes]
    ↓
initial_states_flat: [B*S, state_dim]

# 2. Reshape 到 3D
initial_states_3d: [B, S, state_dim]

# 3. Neural ODE
predictions_3d: [B, S, gene_dim]

# 4. Flatten 回 2D（匹配后续处理）
predictions: [B*S, gene_dim]
```

## 配置文件

在 `neural_ode_config.yaml` 中添加了新参数：

```yaml
model:
  # Neural ODE 参数
  use_neural_ode: true
  ode_hidden_dim: 128
  ode_layers: 3
  time_range: [0.0, 1.0]
  num_time_points: 10
  use_cell_attention: true  # 新增：是否使用细胞间注意力
  num_attention_heads: 8     # 新增：注意力头数
```

## 测试结果

所有维度测试通过：

```
✓ PerturbationODEFunc 3D test passed
  Input:  x=[16, 128, 512], pert_emb=[16, 1280], t=0.5
  Output: dx/dt=[16, 128, 512]

✓ NeuralODEPerturbationModel test passed
  Input:  initial_states=[16, 128, 512], pert_emb=[16, 1280]
  Output: predictions=[16, 128, 18080]
  Trajectory: [10, 16, 128, 512]
```

## 生物学意义

### 为什么 seq_len 很重要？

1. **细胞群体异质性**
   - 同一 perturbation 下的 128 个细胞代表了细胞群体的异质性
   - 通过 attention 机制，模型可以学习群体中的"领导者"和"跟随者"

2. **细胞间通讯**
   - 真实的细胞会通过分泌信号分子相互影响
   - Self-attention 可以捕捉这种细胞间的相互作用

3. **群体效应**
   - 细胞的响应不是独立的，而是受周围细胞影响
   - 3D 架构允许模型学习这种集体行为

### 潜在应用

1. **药物响应预测**
   - 不同细胞对同一药物的响应差异
   - 群体中的敏感细胞和耐药细胞

2. **发育过程建模**
   - 细胞分化过程中的协调演化
   - 组织形成中的细胞间通讯

3. **疾病进展分析**
   - 肿瘤细胞群体的演化
   - 免疫细胞的协同响应

## 下一步计划

1. **训练和评估**
   - 在真实数据集上训练 3D ODE 模型
   - 对比 2D vs 3D 架构的性能

2. **可视化分析**
   - 可视化细胞间注意力权重
   - 分析哪些细胞对群体演化影响最大

3. **超参数调优**
   - attention heads 数量
   - ODE 时间步数
   - 隐藏层维度

## 总结

这次改进将 Neural ODE 从简单的"独立细胞演化"升级为"细胞群体动力学建模"，具有更强的生物学意义和表达能力。通过保留序列结构和引入细胞间注意力，模型现在可以学习更复杂的细胞群体行为模式。
