# 解决 Val/Test 为空的问题

## 🎯 问题原因
其他数据集（jurkat, k562, rpe1, hepg2）**没有 control 样本**（"non-targeting"），因此无法形成 (control, perturbed) pairs，导致 val/test 为 0。

## ✅ 解决方案：自动划分 train 数据

我已经添加了 `--val_split` 和 `--test_split` 参数，可以自动从 train 数据中划分出 val 和 test。

---

## 📝 步骤 1：创建简化的 TOML 配置

**只使用 `competition_train.h5`**（因为它有 control 样本）：

```bash
cat > /data/starter.toml.simple << 'EOF'
[datasets]
competition_train = "/data/competition_train.h5"

[training]
competition_train = "train"
EOF
```

---

## 🚀 步骤 2：使用自动划分训练

```bash
export PYTHONPATH=/workspace:$PYTHONPATH

python -m gnn.cli.train_gnn_simple \
    --data_dir /data \
    --toml_config /data/starter.toml.simple \
    --pert_features /data/ESM2_pert_features.pt \
    --gnn_hidden_dim 128 \
    --gnn_layers 3 \
    --gnn_type gcn \
    --string_confidence 0.4 \
    --batch_size 8 \
    --max_epochs 50 \
    --max_steps 80000 \
    --num_workers 4 \
    --val_split 0.1 \
    --test_split 0.1
```

### 参数说明
- `--val_split 0.1`: 从 train 数据中拿出 10% 作为验证集
- `--test_split 0.1`: 从 train 数据中拿出 10% 作为测试集
- 实际训练集：剩余的 80%

---

## 📊 预期结果

```
Full training dataset loaded with 7110 samples
Auto-splitting train data:
  Val split: 10.0%
  Test split: 10.0%
✅ Auto-split complete:
  Train: 5688 samples (80%)
  Val: 711 samples (10%)
  Test: 711 samples (10%)
```

---

## 🔧 其他可选比例

### 更多验证数据
```bash
--val_split 0.15 --test_split 0.1   # Train: 75%, Val: 15%, Test: 10%
```

### 更多测试数据
```bash
--val_split 0.1 --test_split 0.15   # Train: 75%, Val: 10%, Test: 15%
```

### 只有验证集（不需要测试集）
```bash
--val_split 0.2 --test_split 0.0    # Train: 80%, Val: 20%, Test: 0%
```

---

## ⚡ 快速命令（推荐）

```bash
# 步骤 1: 创建简化 TOML
cat > /data/starter.toml.simple << 'EOF'
[datasets]
competition_train = "/data/competition_train.h5"

[training]
competition_train = "train"
EOF

# 步骤 2: 训练（自动划分 10% val, 10% test）
export PYTHONPATH=/workspace:$PYTHONPATH
python -m gnn.cli.train_gnn_simple \
    --data_dir /data \
    --toml_config /data/starter.toml.simple \
    --pert_features /data/ESM2_pert_features.pt \
    --val_split 0.1 \
    --test_split 0.1 \
    --batch_size 8 \
    --max_epochs 50
```

---

## 🧪 验证是否成功

训练开始后，你应该看到：

```
✅ Full training dataset loaded with 7110 samples
✅ Auto-split complete:
  Train: 5688 samples
  Val: 711 samples
  Test: 711 samples
✅ Model created (GNN enabled: True)
✅ Data module created
✅ Trainer ready
Starting training...
```

✅ **不再有 "Created 0 pairs" 的错误！**

---

## 💡 为什么这样有效？

1. **`competition_train.h5` 有 control 样本**
   - 包含 "non-targeting" 细胞
   - 可以形成 (control, perturbed) pairs

2. **自动划分保证所有 split 都有数据**
   - Train/Val/Test 都来自同一个数据集
   - 所有 split 都能形成 pairs

3. **随机划分确保代表性**
   - `random_seed=42` 保证可重复性
   - 打乱后再划分，避免偏差

---

## 🔄 如果仍然想用其他数据集

如果你坚持使用 jurkat/k562/rpe1/hepg2：

1. **检查这些文件是否有 control 样本**：
   ```bash
   python -c "
   import h5py
   with h5py.File('/data/jurkat.h5', 'r') as f:
       obs = f['obs'][:]
       print('Perturbations:', set(obs['target_gene']))
   "
   ```

2. **如果没有 control，你需要添加**：
   - 从其他数据集复制 "non-targeting" 样本
   - 或者标记某些样本为 control

---

## 📚 代码更新

已更新的文件：
- `gnn/cli/train_gnn_simple.py`: 添加 `--val_split`, `--test_split` 参数
- `gnn/cli/train.py`: `SE_ST_DataModule` 支持自动划分

Commit: 即将提交到 GitHub
