# 修复 Train/Val/Test Split 的问题

## 问题
当前所有数据都分配给了 `train`，导致 `val` 和 `test` 为空。

## 解决方案 1：直接修改 TOML（最简单）

在服务器上运行这个命令直接修改 `/data/starter.toml.working`：

```bash
cat > /data/starter.toml.working << 'EOF'
[datasets]
competition_train = "/data/competition_train.h5"
jurkat = "/data/jurkat.h5"
k562_gwps = "/data/k562_gwps.h5"
rpe1 = "/data/rpe1.h5"
hepg2 = "/data/hepg2.h5"
k562 = "/data/k562.h5"

[training]
competition_train = "train"
k562 = "train"
hepg2 = "train"
jurkat = "val"
k562_gwps = "val"
rpe1 = "test"
EOF
```

### 新的数据划分
- **Train** (3个数据集): competition_train, k562, hepg2
- **Val** (2个数据集): jurkat, k562_gwps
- **Test** (1个数据集): rpe1

## 解决方案 2：使用 Python 脚本

```bash
cd /workspace/gnn
python -m gnn.cli.fix_toml_splits --input /data/starter.toml.working
```

## 验证修改
修改后，查看新配置：

```bash
cat /data/starter.toml.working
```

## 重新训练
修改后重新运行训练：

```bash
python -m gnn.cli.train_gnn_simple \
    --data_dir /data \
    --toml_config /data/starter.toml.working \
    --pert_features /data/ESM2_pert_features.pt \
    --gnn_hidden_dim 128 \
    --gnn_layers 3 \
    --gnn_type gcn \
    --string_confidence 0.4 \
    --batch_size 8 \
    --max_epochs 50 \
    --max_steps 80000 \
    --num_workers 4
```

现在应该会看到：
- ✅ Train: 有数据 (competition_train + k562 + hepg2)
- ✅ Val: 有数据 (jurkat + k562_gwps)
- ✅ Test: 有数据 (rpe1)
