#!/bin/bash

# Neural ODE Training Script
# 修复了导入错误的训练脚本

echo "🚀 开始 Neural ODE 训练..."

# 1. 创建配置文件
cat > /data/neural_ode_starter.toml << 'EOF'
# Neuro-IDE Configuration
[datasets]
replogle_h1 = "/data/{competition_train,k562_gwps,rpe1,jurkat,k562,hepg2}.h5"

[training]
replogle_h1 = "train"

[zeroshot]
#将 hepg2 设为 test
"replogle_h1.hepg2" = "test"
#将 rpe1 设为 val  
"replogle_h1.rpe1" = "val"

[fewshot]
EOF

echo "✅ 已更新/data/neural_ode_starter.toml"

# 2. 安装依赖
echo "📦 安装依赖..."
pip install tomli

# 3. 运行训练
echo "🏃 开始训练..."
cd /workspace/neuro-ide

python neuro-IDE/cli/train_neural_ode_fixed.py \
  --config neuro-IDE/configs/neural_ode_config.yaml \
  --data.toml_config_path="/data/neural_ode_starter.toml" \
  --data.perturbation_features_file="/data/ESM2_pert_features.pt" \
  --data.num_workers=4 \
  --training.max_steps=80000 \
  --training.batch_size=8 \
  --training.val_check_interval=100 \
  --model.input_dim=18080 \
  --model.use_neural_ode=true \
  --model.ode_hidden_dim=128 \
  --model.ode_layers=3 \
  --model.time_range="[0.0, 1.0]" \
  --model.num_time_points=10

echo "🎉 训练完成！"
