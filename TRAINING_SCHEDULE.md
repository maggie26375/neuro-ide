# 🚀 SE-ST-Combined 训练计划

## 推荐训练步数：**40,000 - 50,000 steps**

### 理由：
1. ✅ SE encoder (600M) 已冻结 → 只训练 ST 部分 (48.9M params)
2. ✅ ST 部分复杂度与 STATE 相当
3. ✅ 输入是 SE embeddings（已经过预训练）→ 可能收敛更快
4. ⚠️ 但新的输入空间可能需要额外的适应步数

---

## 完整训练命令（推荐）

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
  ++training.log_every_n_steps=50 \
  ++training.early_stopping=true \
  ++training.patience=10 \
  ++model.kwargs.input_dim=18080 \
  ++model.kwargs.hidden_dim=512 \
  ++model.kwargs.output_dim=18080 \
  ++model.kwargs.pert_dim=5120 \
  ++model.kwargs.se_model_path="SE-600M" \
  ++model.kwargs.se_checkpoint_path="SE-600M/se600m_epoch15.ckpt" \
  ++output_dir="competition" \
  ++name="se_st_combined_40k"
```

---

## 关键参数说明

| 参数 | 值 | 说明 |
|------|-----|------|
| `max_steps` | 40000 | 总训练步数（与 STATE 对齐） |
| `ckpt_every_n_steps` | 5000 | 每 5k 步保存一次（8 个 checkpoints） |
| `val_check_interval` | 500 | 每 500 步验证一次 |
| `batch_size` | 16 | 根据 GPU 内存调整（8-32） |
| `lr` | 1e-4 | 学习率（STATE 默认） |
| `early_stopping` | true | 自动停止（防止过拟合） |
| `patience` | 10 | 验证 loss 10 次不下降则停止 |
| `num_workers` | 8 | 数据加载进程数 |

---

## 训练时间估算

基于你的测试运行（1000 steps）：

- **1000 steps**: ~5 分钟
- **5000 steps**: ~25 分钟
- **20000 steps**: ~1.5 小时
- **40000 steps**: ~3 小时

**实际时间取决于**：
- GPU 型号（A100 / V100 / T4）
- Batch size（16 vs 32）
- 数据加载速度

---

## 监控指标

### 🟢 训练健康的信号：
```
Epoch 0:  50%|█████     | 500/1000 [02:30<02:30, 3.3it/s, loss=2.5, val_loss=2.8]
Epoch 0: 100%|██████████| 1000/1000 [05:00<00:00, 3.3it/s, loss=2.1, val_loss=2.3]
Epoch 1:  50%|█████     | 500/1000 [02:30<02:30, 3.3it/s, loss=1.8, val_loss=2.0]
```

✅ `loss` 持续下降  
✅ `val_loss` 跟随下降  
✅ `it/s` 稳定（3-5 iterations/second）

---

### 🔴 需要调整的信号：

**1. 过拟合**：
```
loss=1.2, val_loss=3.5  # val_loss >> loss
```
→ 降低模型复杂度或增加正则化

**2. 欠拟合**：
```
loss=4.5, val_loss=4.6  # 都很高
```
→ 增加模型容量或训练更多步数

**3. 训练不稳定**：
```
loss=2.1 → 5.3 → 1.8 → 4.2  # 震荡
```
→ 降低学习率（1e-5）或增加 gradient clipping

**4. 训练太慢**：
```
it/s < 1  # 每秒少于 1 iteration
```
→ 减少 `num_workers` 或检查数据加载

---

## 分阶段训练策略（可选）

如果不确定 40k 是否足够，可以分阶段：

### Stage 1: Quick Test (5k steps)
```bash
se-st-train \
  ++training.max_steps=5000 \
  ++name="stage1_quick_test"
```

**目标**: 验证模型能收敛

---

### Stage 2: Medium Run (20k steps)
```bash
se-st-train \
  ++training.max_steps=20000 \
  ++model.checkpoint="competition/stage1_quick_test/checkpoint-step-5000.ckpt" \
  ++name="stage2_medium_run"
```

**目标**: 观察 loss 下降趋势

---

### Stage 3: Full Training (40k steps)
```bash
se-st-train \
  ++training.max_steps=40000 \
  ++model.checkpoint="competition/stage2_medium_run/checkpoint-step-20000.ckpt" \
  ++name="stage3_full_training"
```

**目标**: 充分训练到收敛

---

### Stage 4: Extended Training (如果需要)
```bash
se-st-train \
  ++training.max_steps=60000 \
  ++model.checkpoint="competition/stage3_full_training/checkpoint-step-40000.ckpt" \
  ++name="stage4_extended"
```

**条件**: 40k 步时 loss 仍在下降

---

## 对比：你的模型 vs STATE

| 维度 | STATE | SE-ST-Combined |
|------|-------|----------------|
| **输入维度** | 18080 (genes) | 18080 (genes) |
| **中间表示** | 直接使用 | SE embeddings (512D) |
| **Trainable Params** | ~50M | 48.9M |
| **冻结部分** | 无 | SE encoder (600M) |
| **训练步数** | 40,000 | **推荐 40,000-50,000** |

---

## FAQ

### Q1: 为什么不直接训练 100k 步？
**A**: 
- 💰 时间成本：100k 步 ~7.5 小时
- 📉 收益递减：40k 步后 loss 可能已经收敛
- ⚠️ 过拟合风险：训练太久可能伤害泛化能力

### Q2: 如果 40k 步还没收敛怎么办？
**A**: 
```bash
# 从 checkpoint 继续训练
se-st-train \
  ++training.max_steps=60000 \
  ++model.checkpoint="competition/se_st_combined_40k/checkpoint-step-40000.ckpt" \
  ++name="se_st_combined_60k"
```

### Q3: 如何知道模型已经收敛？
**A**: 观察这些信号：
1. ✅ Loss 曲线平台期（连续 5k 步变化 < 1%）
2. ✅ Validation loss 不再下降
3. ✅ Early stopping 触发

### Q4: 我的 GPU 内存不够怎么办？
**A**: 调整这些参数：
```bash
++training.batch_size=8        # 降低 batch size
++training.gradient_clip_val=1.0  # 添加 gradient clipping
++training.accumulate_grad_batches=2  # 梯度累积
```

---

## 最终建议

**对于你的 SE-ST-Combined 模型，我建议：**

1. 🎯 **首次完整训练**：40,000 steps
2. 📊 **监控指标**：每 500 steps 看 val_loss
3. ⏱️ **预计时间**：~3 小时（A100）
4. 💾 **保存策略**：每 5k 步保存 checkpoint
5. 🛑 **Early stopping**：patience=10（5k steps）

**如果 40k 步时：**
- ✅ Loss 已收敛 → 完成训练
- ⚠️ 仍在下降 → 继续到 50k-60k
- ❌ 过拟合 → 回退到更早的 checkpoint

**开始训练吧！** 🚀

