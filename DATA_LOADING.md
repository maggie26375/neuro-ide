# 数据加载说明

## ✅ 已实现完整的数据加载器！

现在你可以直接使用 `uv run se-st-train` 来训练模型了！

## 📁 数据格式要求

### 1. TOML 配置文件 (`data/starter.toml`)

```toml
[datasets]
replogle_h1 = "/path/to/data/{competition_train,k562_gwps,rpe1,jurkat,k562,hepg2}.h5"

[training]
replogle_h1 = "train"

[zeroshot]
"replogle_h1.hepg2" = "test"

[fewshot]
# 可选: 指定特定样本用于 few-shot 学习
```

### 2. H5 数据文件

每个 H5 文件应该包含：
- `X`: 表达矩阵 (n_cells × n_genes)
- `obs`: 元数据 (包含 perturbation, batch, cell_type 等)
- `var`: 基因信息 (可选)

### 3. 扰动特征文件 (`data/ESM2_pert_features.pt`)

PyTorch tensor 字典，格式：
```python
{
    "gene_name_1": torch.Tensor([...]),  # ESM2 embedding
    "gene_name_2": torch.Tensor([...]),
    ...
}
```

## 🚀 训练命令

### 完整命令（在 vast.ai 上运行）

```bash
uv run se-st-train \
  ++data.kwargs.toml_config_path="data/starter.toml" \
  ++data.kwargs.perturbation_features_file="data/ESM2_pert_features.pt" \
  ++data.kwargs.num_workers=8 \
  ++data.kwargs.batch_col="batch_var" \
  ++data.kwargs.pert_col="target_gene" \
  ++data.kwargs.cell_type_key="cell_type" \
  ++data.kwargs.control_pert="non-targeting" \
  ++training.max_steps=40000 \
  ++training.ckpt_every_n_steps=20000 \
  ++training.batch_size=16 \
  ++model.kwargs.se_model_path="SE-600M" \
  ++model.kwargs.se_checkpoint_path="SE-600M/se600m_epoch15.ckpt" \
  ++output_dir="competition" \
  ++name="first_run"
```

## 🔧 数据加载流程

1. **读取 TOML 配置**：解析数据集路径和训练/验证/测试分割
2. **展开文件模式**：`{a,b,c}.h5` → `[a.h5, b.h5, c.h5]`
3. **加载 H5 文件**：读取表达矩阵和元数据
4. **加载扰动特征**：从 ESM2_pert_features.pt 加载蛋白质嵌入
5. **创建数据集**：根据 split (train/val/test) 过滤数据
6. **创建 DataLoader**：批量加载数据用于训练

## 📊 数据分割策略

- **Training**: 所有未在 `zeroshot` 中标记为 test/val 的数据
- **Validation**: 标记为 "val" 的数据（如果没有，使用 test 数据）
- **Test**: 标记为 "test" 的数据（零样本学习）

根据你的 TOML：
- **Train**: competition_train, k562_gwps, rpe1, jurkat, k562
- **Test**: hepg2 (零样本细胞类型)

## 🐛 故障排查

### 问题 1: "No training batches"
**原因**: 数据文件路径不正确或文件不存在

**解决方案**:
```bash
# 检查数据文件
ls -la data/

# 确保路径正确
# TOML 中的路径会自动替换为 data/ 目录
```

### 问题 2: "No H5 files found"
**原因**: TOML 中的文件模式无法匹配实际文件

**解决方案**:
```bash
# 检查 TOML 中的模式
# 例如: {k562,hepg2}.h5 应该匹配 k562.h5 和 hepg2.h5

# 确保文件存在
ls -la data/*.h5
```

### 问题 3: KeyError in H5 file
**原因**: H5 文件结构不符合预期

**解决方案**:
```python
# 检查 H5 文件内容
import h5py
with h5py.File("data/k562.h5", "r") as f:
    print(list(f.keys()))  # 查看顶层键
```

## 📝 自定义数据加载

如果需要修改数据加载逻辑，编辑：
```
se_st_combined/data/perturbation_dataset.py
```

主要方法：
- `_load_toml_config()`: 解析 TOML 文件
- `_load_h5_data()`: 加载 H5 数据
- `_load_single_h5()`: 处理单个 H5 文件
- `__getitem__()`: 返回单个样本

## 📦 依赖项

新增依赖（已添加到 requirements.txt）：
- `h5py>=3.7.0`: H5 文件读取
- `tomli>=2.0.0`: TOML 文件解析

安装：
```bash
pip install h5py tomli
```

## ✅ 测试数据加载

快速测试数据加载器：
```python
from se_st_combined.data import PerturbationDataset

# 创建数据集
dataset = PerturbationDataset(
    toml_config_path="data/starter.toml",
    perturbation_features_file="data/ESM2_pert_features.pt",
    split="train"
)

print(f"Loaded {len(dataset)} samples")

# 查看第一个样本
sample = dataset[0]
print(sample.keys())
```

## 🎯 下一步

1. **在 vast.ai 上运行**：
   ```bash
   cd /workspace/se-st-combined
   uv run se-st-train ++data.kwargs.toml_config_path="data/starter.toml" ...
   ```

2. **监控训练**：观察日志中的数据加载信息
   - "Loaded X samples for split 'train'"
   - "Training dataset created with X samples"

3. **调试**：如果遇到问题，查看详细日志
   - 数据文件是否找到
   - H5 文件是否正确加载
   - 扰动嵌入是否加载成功

祝训练顺利！🚀

