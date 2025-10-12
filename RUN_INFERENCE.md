# 🔮 SE+ST Combined Model - Inference Guide

完整的推理流程，从 checkpoint 到预测结果！

---

## 📋 **准备工作**

### 1️⃣ **在 vast.ai 上拉取最新代码**

```bash
cd /workspace/se-st-combined
git pull origin main
pip install -e .
```

### 2️⃣ **确认文件存在**

```bash
# 检查 checkpoint
ls -lh competition/se_st_combined_40k_v2/final_model.ckpt

# 检查验证数据（需要从 Kaggle 下载）
ls -lh /data/competition_val_template.h5ad

# 检查 ESM2 features
ls -lh /data/ESM2_pert_features.pt

# 检查 SE 模型
ls -lh SE-600M/se600m_epoch15.ckpt
```

---

## 🚀 **运行推理**

### **完整命令**

```bash
se-st-infer \
  --checkpoint competition/se_st_combined_40k_v2/final_model.ckpt \
  --adata /data/competition_val_template.h5ad \
  --output competition/prediction.h5ad \
  --pert-col target_gene \
  --se-model-path SE-600M \
  --perturbation-features /data/ESM2_pert_features.pt \
  --batch-size 16 \
  --device cuda
```

### **参数说明**

| 参数 | 说明 | 示例 |
|------|------|------|
| `--checkpoint` | 训练好的模型 checkpoint | `final_model.ckpt` |
| `--adata` | 输入数据（H5AD 格式） | `competition_val_template.h5ad` |
| `--output` | 输出预测结果（H5AD 格式） | `prediction.h5ad` |
| `--pert-col` | AnnData.obs 中的 perturbation 列名 | `target_gene` |
| `--se-model-path` | SE 模型目录 | `SE-600M` |
| `--perturbation-features` | ESM2 perturbation embeddings | `ESM2_pert_features.pt` |
| `--batch-size` | 推理 batch size（可调） | `16` |
| `--device` | 运行设备 | `cuda` 或 `cpu` |

---

## 📊 **推理输出**

运行成功后，你会看到：

```
[INFO] Loading model from checkpoint: competition/se_st_combined_40k_v2/final_model.ckpt
[INFO] Found hyperparameters in checkpoint: ['input_dim', 'hidden_dim', ...]
[INFO] Model weights loaded successfully
[INFO] Loading perturbation features from /data/ESM2_pert_features.pt
[INFO] Loaded 1234 perturbation features
[INFO] Loading input data from /data/competition_val_template.h5ad
[INFO] Loaded 10000 cells x 18080 genes
[INFO] Running inference on 10000 cells
[INFO] Found 50 unique perturbations
Processing perturbations: 100%|████████| 50/50 [00:30<00:00]
[INFO] Inference completed! Predictions shape: (10000, 18080)
[INFO] Saving predictions to competition/prediction.h5ad
✅ Inference completed successfully!
```

---

## 📦 **提交到比赛**

### **1. 准备提交文件**

```bash
# 安装 zstd（如果还没安装）
sudo apt install -y zstd

# 运行 cell-eval prep
uv tool run --from git+https://github.com/ArcInstitute/cell-eval@main cell-eval prep \
  -i competition/prediction.h5ad \
  -g /data/gene_names.csv
```

这会生成 `submission.vcc` 文件（压缩格式）。

### **2. 下载提交文件**

```bash
# 在 vast.ai 上查看文件
ls -lh competition/*.vcc

# 使用 scp 或 rsync 下载到本地
# 或者在 JupyterLab 中直接下载
```

### **3. 上传到 Kaggle**

前往 Virtual Cell Challenge 的提交页面，上传 `.vcc` 文件。

---

## 🔧 **故障排除**

### **Error: Checkpoint not found**

```bash
# 检查 checkpoint 路径
ls -lh competition/se_st_combined_40k_v2/

# 可能需要使用其他 checkpoint
se-st-infer \
  --checkpoint competition/se_st_combined_40k_v2/checkpoints/step=step=40000-val_loss=val_loss=39.3924.ckpt \
  # ... 其他参数
```

### **Error: AnnData file not found**

```bash
# 需要从 Kaggle 下载验证数据
# 或者使用测试数据
--adata /data/competition_test.h5ad
```

### **CUDA Out of Memory**

```bash
# 减小 batch size
--batch-size 8
# 或者使用 CPU
--device cpu
```

### **No embedding found for perturbation**

推理脚本会自动处理缺失的 perturbation embeddings（使用零向量）。

---

## 🎯 **多个 Checkpoint 推理（Ensemble）**

如果你训练了多个模型，可以分别推理然后平均：

```bash
# 推理 checkpoint 1
se-st-infer \
  --checkpoint competition/run1/final_model.ckpt \
  --output competition/pred1.h5ad \
  # ... 其他参数

# 推理 checkpoint 2
se-st-infer \
  --checkpoint competition/run2/final_model.ckpt \
  --output competition/pred2.h5ad \
  # ... 其他参数

# 在 Python 中平均预测结果
python -c "
import anndata
pred1 = anndata.read_h5ad('competition/pred1.h5ad')
pred2 = anndata.read_h5ad('competition/pred2.h5ad')
ensemble = pred1.copy()
ensemble.X = (pred1.X + pred2.X) / 2
ensemble.write_h5ad('competition/ensemble_pred.h5ad')
"
```

---

## 📈 **预期性能**

根据训练结果（val_loss = 39.39），你应该期待：

1. **良好的泛化能力**：模型在 held-out cell types 上表现稳定
2. **稳定的预测**：不同细胞的相同 perturbation 预测一致
3. **生物学合理性**：预测的基因表达变化符合生物学预期

---

## 🚀 **快速命令（Copy & Paste）**

```bash
# Step 1: 拉取代码并安装
cd /workspace/se-st-combined
git pull origin main
pip install -e .

# Step 2: 运行推理
se-st-infer \
  --checkpoint competition/se_st_combined_40k_v2/final_model.ckpt \
  --adata /data/competition_val_template.h5ad \
  --output competition/prediction.h5ad \
  --pert-col target_gene \
  --se-model-path SE-600M \
  --perturbation-features /data/ESM2_pert_features.pt \
  --batch-size 16 \
  --device cuda

# Step 3: 准备提交
uv tool run --from git+https://github.com/ArcInstitute/cell-eval@main cell-eval prep \
  -i competition/prediction.h5ad \
  -g /data/gene_names.csv

# Step 4: 下载并提交
ls -lh competition/*.vcc
```

---

## 💡 **提示**

1. **第一次推理**：建议先用小的 batch size 测试（`--batch-size 4`）
2. **监控内存**：使用 `nvidia-smi` 查看 GPU 内存使用
3. **备份预测**：推理完成后立即备份 `.h5ad` 文件
4. **验证输出**：检查预测的 shape 和数值范围是否合理

---

**祝你在 Virtual Cell Challenge 中取得好成绩！** 🏆

