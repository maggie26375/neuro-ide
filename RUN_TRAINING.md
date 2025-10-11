# 🚀 运行训练完整流程

## Step 1: 在远程服务器上更新代码

```bash
cd /workspace/se-st-combined
git pull origin main
pip install -e .
```

---

## Step 2: 快速测试数据加载

```bash
python3 << 'EOF'
from se_st_combined.data import PerturbationDataset

dataset = PerturbationDataset(
    toml_config_path="data/starter.toml",
    perturbation_features_file="/data/ESM2_pert_features.pt",
    split="train",
)

print(f"✅ Created {len(dataset)} sentence specs")

if len(dataset) > 0:
    sample = dataset[0]
    print(f"\n📦 First sample:")
    print(f"  ctrl_cell_emb: {sample['ctrl_cell_emb'].shape}")
    print(f"  pert_cell_emb: {sample['pert_cell_emb'].shape}")
    print(f"  pert_emb: {sample['pert_emb'].shape}")
    print(f"  Perturbation: {sample['perturbation']}")
    print(f"  Cell type: {sample['cell_type']}")
else:
    print("❌ No data! Check TOML config and H5 files.")
EOF
```

**预期输出**：
```
✅ Created XXXX sentence specs

📦 First sample:
  ctrl_cell_emb: torch.Size([128, 18080])
  pert_cell_emb: torch.Size([128, 18080])
  pert_emb: torch.Size([128, 5120])
  Perturbation: ARPC2
  Cell type: K562
```

---

## Step 3: 开始训练

```bash
se-st-train \
  ++data.kwargs.toml_config_path="data/starter.toml" \
  ++data.kwargs.perturbation_features_file="/data/ESM2_pert_features.pt" \
  ++data.kwargs.num_workers=4 \
  ++training.max_steps=1000 \
  ++training.batch_size=8 \
  ++training.lr=1e-4 \
  ++training.val_check_interval=100 \
  ++model.kwargs.input_dim=18080 \
  ++model.kwargs.hidden_dim=512 \
  ++model.kwargs.output_dim=18080 \
  ++model.kwargs.pert_dim=5120 \
  ++model.kwargs.se_model_path="SE-600M" \
  ++model.kwargs.se_checkpoint_path="SE-600M/se600m_epoch15.ckpt" \
  ++output_dir="competition" \
  ++name="first_test"
```

**参数说明**：
- `input_dim=18080`：你的数据的基因数（从前面的 shape 可以看出）
- `output_dim=18080`：输出维度（与输入相同）
- `pert_dim=5120`：ESM2 embedding 维度
- `batch_size=8`：先用小 batch 测试
- `num_workers=4`：4个数据加载进程
- `max_steps=1000`：测试 1000 步
- `val_check_interval=100`：每 100 步验证一次

---

## Step 4: 如果遇到错误

### 错误 1: `ValueError: num_samples should be a positive integer value, but got num_samples=0`
**原因**：数据加载失败，没有创建任何 sentence specs

**检查**：
```bash
# 检查 TOML 配置
cat data/starter.toml

# 检查 H5 文件是否存在
ls -lh /data/*.h5

# 运行 Step 2 的测试脚本
```

---

### 错误 2: `RuntimeError: mat1 and mat2 shapes cannot be multiplied`
**原因**：`input_dim` 设置错误

**解决**：从 Step 2 的输出中找到正确的 `ctrl_cell_emb` 的第二维，例如：
```
ctrl_cell_emb: torch.Size([128, 18080])  # input_dim=18080
```

---

### 错误 3: `KeyError: 'pert_emb'`
**原因**：模型期望的键名与数据加载器不匹配

**已修复**：最新代码已经修复了这个问题（`pert_embedding` → `pert_emb`）

---

### 错误 4: `FileNotFoundError: data/ESM2_pert_features.pt`
**原因**：路径错误

**解决**：使用绝对路径
```bash
++data.kwargs.perturbation_features_file="/data/ESM2_pert_features.pt"
```

---

## Step 5: 完整训练（40000 步）

如果 Step 3 的测试运行成功，改为完整训练：

```bash
se-st-train \
  ++data.kwargs.toml_config_path="data/starter.toml" \
  ++data.kwargs.perturbation_features_file="/data/ESM2_pert_features.pt" \
  ++data.kwargs.num_workers=8 \
  ++training.max_steps=40000 \
  ++training.ckpt_every_n_steps=5000 \
  ++training.batch_size=16 \
  ++training.lr=1e-4 \
  ++training.val_check_interval=500 \
  ++model.kwargs.input_dim=18080 \
  ++model.kwargs.hidden_dim=512 \
  ++model.kwargs.output_dim=18080 \
  ++model.kwargs.pert_dim=5120 \
  ++model.kwargs.se_model_path="SE-600M" \
  ++model.kwargs.se_checkpoint_path="SE-600M/se600m_epoch15.ckpt" \
  ++output_dir="competition" \
  ++name="se_st_combined_run1"
```

---

## 📊 监控训练

训练应该会输出：
```
Epoch 0:  10%|█         | 100/1000 [XX:XX<XX:XX, X.XXit/s, loss=X.XX]
```

**关键指标**：
- `loss`：应该逐渐下降
- `val_loss`：验证集损失
- `it/s`：迭代速度（iterations per second）

---

## 🎯 预期结果

1. ✅ 数据加载成功（不再是 `num_samples=0`）
2. ✅ 模型初始化成功（SE + ST 模型）
3. ✅ 训练循环开始（loss 开始下降）

如果出现新的错误，把完整的错误信息发给我！

