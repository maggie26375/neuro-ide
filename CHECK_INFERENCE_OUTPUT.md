# 检查推理输出

## 在 Colab 中运行以下代码验证结果：

```python
import os
import anndata
import numpy as np

# 检查输出文件
output_path = "/content/drive/MyDrive/prediction_hvg_scanpy.h5ad"

if os.path.exists(output_path):
    print(f"✅ 输出文件已生成！")
    print(f"   大小: {os.path.getsize(output_path) / (1024**2):.2f} MB")
    
    # 读取数据
    adata = anndata.read_h5ad(output_path)
    print(f"\n📊 输出数据形状: {adata.shape}")
    print(f"   - {adata.n_obs} observations (cell-perturbation pairs)")
    print(f"   - {adata.n_vars} genes")
    
    # 检查预测值
    print(f"\n🔍 预测值统计:")
    if hasattr(adata.X, 'toarray'):
        X = adata.X.toarray()
    else:
        X = adata.X
    
    print(f"   - Min: {X.min():.4f}")
    print(f"   - Max: {X.max():.4f}")
    print(f"   - Mean: {X.mean():.4f}")
    print(f"   - Std: {X.std():.4f}")
    print(f"   - Non-zero values: {(X != 0).sum()} / {X.size}")
    
    # 检查 obs 信息
    print(f"\n📝 Observation 信息:")
    print(f"   - Columns: {adata.obs.columns.tolist()}")
    if 'perturbation' in adata.obs.columns:
        print(f"   - Unique perturbations: {adata.obs['perturbation'].nunique()}")
    
    print("\n✅ 推理完成！文件可以提交到比赛。")
else:
    print("❌ 输出文件未生成！")
```

---

## ⚠️ 维度不匹配问题说明

### 问题：
- **训练时模型配置**: `pert_dim = 1280`
- **实际 embedding 文件**: `ESM2_pert_features.pt` 是 **5120** 维
- **当前处理**: 推理脚本自动**截断**前 1280 维

### 影响：
- ✅ **能跑通**: 推理可以完成
- ⚠️ **性能损失**: 截断了 embedding 信息，可能影响预测质量

### 解决方案：

#### 临时方案（当前可用）：
- 使用当前输出提交比赛，虽然不是最优但应该有合理结果

#### 正确方案（推荐）：
重新训练模型，确保 `pert_dim=5120`：

```bash
se-st-train \
  ++data.kwargs.toml_config_path="data/starter.toml" \
  ++data.kwargs.perturbation_features_file="/data/ESM2_pert_features.pt" \
  ++training.max_steps=40000 \
  ++training.batch_size=8 \
  ++model.kwargs.pert_dim=5120 \        # ⬅️ 确保这里是 5120
  ++model.kwargs.input_dim=18080 \
  ++model.kwargs.output_dim=18080 \
  # ... 其他参数
```

---

## 🎯 下一步：

1. **先验证当前输出** - 运行上面的检查代码
2. **提交当前结果** - 看看 baseline 性能如何
3. **重新训练（可选）** - 如果需要提升性能，使用正确的 `pert_dim=5120` 重新训练



