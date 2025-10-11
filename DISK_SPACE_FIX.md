# 💾 Disk Space Issue & Fix

## ❌ 问题：训练时磁盘空间不足

```
OSError: [Errno 28] No space left on device
```

---

## 🔍 根本原因

### **SE-ST-Combined vs STATE 的 Checkpoint 大小对比**

| 模型 | Trainable | Frozen | **Checkpoint 大小** |
|------|-----------|--------|-------------------|
| **STATE** | 50M params | 0 | **~200 MB** |
| **SE-ST-Combined（你的）** | 48.9M params | **600M (SE)** | **~2.5 GB** |

### **问题详解**

1. **Lightning 默认行为**：
   - 每次验证（`val_check_interval`）保存 `last.ckpt`
   - 保存 top-k 最好的 checkpoints
   - **保存整个模型状态**（包括冻结的参数！）

2. **你的训练设置**：
   - `val_check_interval=250` steps
   - `max_steps=40000`
   - 40000 / 250 = **160 次验证**
   - 160 × 2.5 GB = **400 GB 磁盘空间！**

3. **STATE 为什么没问题？**
   - STATE 没有冻结的大模型
   - Checkpoint 只有 200 MB
   - 160 × 200 MB = 32 GB（可接受）

---

## ✅ 解决方案

### **修复 1: 不保存 `last.ckpt`（已应用）**

修改了 `se_st_combined/cli/train.py`：

```python
checkpoint_callback = ModelCheckpoint(
    save_last=False,  # ← 不在每次验证时保存 last.ckpt
    save_on_train_epoch_end=False,  # 只在验证时保存
    save_top_k=3,  # 只保存最好的 3 个
)
```

**效果**：
- 磁盘使用：400 GB → **~7.5 GB**（top-3 only）
- 仍然每 `ckpt_every_n_steps` (5000) 保存一次（用于恢复训练）

---

### **修复 2: 清理磁盘空间**

```bash
# 1. 清理旧的 checkpoints
rm -rf competition/*/checkpoints/*.ckpt
rm -rf competition/*/last.ckpt

# 2. 清理 pip 缓存
pip cache purge

# 3. 清理 PyTorch 缓存
rm -rf ~/.cache/torch

# 4. 清理临时文件
rm -rf /tmp/*

# 5. 检查磁盘使用
df -h
```

---

### **修复 3: 调整训练参数（可选）**

如果磁盘空间仍然紧张，可以：

#### **A. 减少保存的 checkpoint 数量**
```bash
++save_top_k=1  # 只保存最好的 1 个（而非 3 个）
```

#### **B. 增加保存间隔**
```bash
++training.ckpt_every_n_steps=10000  # 每 10k 步保存（而非 5k）
```

#### **C. 使用更大的验证间隔**
```bash
++training.val_check_interval=490  # 每个 epoch 验证一次
```

---

## 📊 完整训练命令（带优化）

```bash
# 先清理磁盘
rm -rf competition/*/checkpoints/*.ckpt
pip cache purge

# 更新代码
cd /workspace/se-st-combined
git pull origin main
pip install -e .

# 重新训练（使用修复后的配置）
se-st-train \
  ++data.kwargs.toml_config_path="data/starter.toml" \
  ++data.kwargs.perturbation_features_file="/data/ESM2_pert_features.pt" \
  ++data.kwargs.num_workers=8 \
  ++training.max_steps=40000 \
  ++training.ckpt_every_n_steps=5000 \
  ++training.batch_size=16 \
  ++training.lr=1e-4 \
  ++training.val_check_interval=250 \
  ++training.log_every_n_steps=50 \
  ++save_top_k=2 \
  ++model.kwargs.input_dim=18080 \
  ++model.kwargs.hidden_dim=512 \
  ++model.kwargs.output_dim=18080 \
  ++model.kwargs.pert_dim=5120 \
  ++model.kwargs.se_model_path="SE-600M" \
  ++model.kwargs.se_checkpoint_path="SE-600M/se600m_epoch15.ckpt" \
  ++output_dir="competition" \
  ++name="se_st_combined_40k_fixed"
```

**预期磁盘使用**：
- Top-2 checkpoints: 5 GB
- 8 periodic checkpoints (每 5k 步): 20 GB
- **总计**: ~25 GB（可控）

---

## 🔮 未来优化（可选）

### **Option 1: 只保存 trainable parameters**

理论上可以在保存时排除冻结的 SE encoder，但需要自定义 `on_save_checkpoint` 钩子：

```python
def on_save_checkpoint(self, checkpoint):
    # 移除冻结的 SE encoder 参数
    state_dict = checkpoint['state_dict']
    filtered = {k: v for k, v in state_dict.items() 
                if not k.startswith('se_model.')}
    checkpoint['state_dict'] = filtered
    return checkpoint
```

**问题**：加载时需要重新加载 SE encoder，增加复杂度。

---

### **Option 2: 使用外部存储**

如果有云存储（S3/GCS），可以配置 Lightning 直接保存到云端：

```python
checkpoint_callback = ModelCheckpoint(
    dirpath="s3://your-bucket/checkpoints",
    ...
)
```

---

## 📝 监控磁盘使用

训练时定期检查：

```bash
# 每隔 10 分钟检查一次
watch -n 600 df -h

# 或者在后台监控
while true; do 
    df -h | grep '/dev/sda1'
    sleep 600
done
```

---

## ❓ FAQ

### Q1: 为什么不用 `save_weights_only=True`？
**A**: 即使使用 `save_weights_only=True`，Lightning 仍会保存所有参数（包括冻结的）。你需要自定义保存逻辑才能真正排除冻结参数。

### Q2: 如果训练中断了怎么办？
**A**: 每 5000 steps 仍会保存一个 checkpoint，可以从最近的恢复：
```bash
++model.checkpoint="competition/se_st_combined_40k/checkpoints/step=35000-*.ckpt"
```

### Q3: STATE 是怎么处理的？
**A**: STATE 没有冻结的大模型，所以每个 checkpoint 只有 200 MB，不会遇到这个问题。

### Q4: 能不能完全不保存 checkpoint？
**A**: 不推荐！如果训练中断，你会失去所有进度。至少保存 top-1 或 periodic checkpoints。

---

## 🎯 总结

| 策略 | 磁盘节省 | 风险 |
|------|---------|------|
| **不保存 `last.ckpt`** | 95% (400GB → 20GB) | ✅ 低（仍有 top-k 和 periodic） |
| **`save_top_k=1`** | 额外 2.5 GB | ⚠️ 中（只有 1 个最好的） |
| **增加 `ckpt_every_n_steps`** | 额外 10+ GB | ⚠️ 中（恢复点更少） |
| **自定义保存逻辑** | 90% (2.5GB → 200MB) | ⚠️ 高（加载复杂） |

**推荐**：使用当前修复（不保存 `last.ckpt`）+ `save_top_k=2-3` + `ckpt_every_n_steps=5000`。

这样既能节省磁盘空间，又能保证训练的鲁棒性。

