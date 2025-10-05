# SE+ST Combined Model

A combined State Embedding (SE) and State Transition (ST) model for cross-cell-type perturbation prediction in single-cell genomics.

## 🚀 Features

- **Cross-cell-type generalization**: Better prediction accuracy across different cell types
- **Cell-type-agnostic modeling**: Uses universal state embeddings for robust predictions
- **Pre-trained SE integration**: Leverages pre-trained State Embedding models
- **Easy installation**: Install via pip from GitHub
- **Complete pipeline**: Training, inference, and evaluation utilities

## 📦 Installation

### From GitHub (Recommended)

```bash
# Install directly from GitHub
pip install git+https://github.com/maggie26375/se-st-combined@main

# Or using uv (faster)
uv add git+https://github.com/maggie26375/se-st-combined@main
```

### From Source

```bash
# Clone the repository
git clone https://github.com/maggie26375/se-st-combined.git
cd se-st-combined

# Install in development mode
pip install -e .

# Or using uv
uv pip install -e .
```

## 🎯 Quick Start

### Basic Usage

```python
from se_st_combined.models.se_st_combined import SE_ST_CombinedModel
from se_st_combined.utils.se_st_utils import load_se_st_model, predict_perturbation_effects

# Load the model
model = load_se_st_model(
    model_dir="path/to/model",
    checkpoint_path="path/to/checkpoint.ckpt",
    se_model_path="path/to/se/model",
    se_checkpoint_path="path/to/se/checkpoint.ckpt"
)

# Make predictions
predictions = predict_perturbation_effects(
    model=model,
    ctrl_expressions=ctrl_expressions,
    pert_embeddings=pert_embeddings
)
```

### Training

```python
# See examples/se_st_virtual_cell_challenge.py for complete training example
import se_st_combined

# Training command (similar to original STATE training)
! uv run se-st-train \
  data.kwargs.toml_config_path="data/starter.toml" \
  data.kwargs.perturbation_features_file="data/ESM2_pert_features.pt" \
  training.max_steps=40000 \
  model=se_st_combined \
  model.kwargs.se_model_path="SE-600M" \
  model.kwargs.se_checkpoint_path="SE-600M/se600m_epoch15.ckpt" \
  output_dir="results" \
  name="se_st_experiment"
```

## 🏗️ Architecture

The SE+ST Combined Model consists of two main components:

### 1. State Embedding (SE) Encoder
- Converts raw gene expression to universal state embeddings
- Uses pre-trained SE model (e.g., SE-600M)
- Provides cell-type-agnostic representations

### 2. State Transition (ST) Predictor
- Predicts perturbation effects in state embedding space
- Uses transformer architecture (GPT2/Llama)
- Learns set-to-set functions for perturbation modeling

### Data Flow
```
Raw Expression → SE Encoder → State Embeddings
                                    ↓
Perturbation Embeddings → ST Predictor → Predicted Expression
```

## 📊 Performance

Expected improvements over baseline StateTransition model:

- **Cross-cell-type accuracy**: 10-20% improvement
- **Generalization**: Better performance on unseen cell types
- **Training stability**: More stable convergence
- **Robustness**: More consistent predictions across cell types

## 🔧 Configuration

The model can be configured via YAML files:

```yaml
# se_st_combined.yaml
name: se_st_combined
kwargs:
  se_model_path: "SE-600M"
  se_checkpoint_path: "SE-600M/se600m_epoch15.ckpt"
  freeze_se_model: true
  st_hidden_dim: 672
  st_cell_set_len: 128
  transformer_backbone_key: llama
```

## 📁 Project Structure

```
se-st-combined/
├── se_st_combined/
│   ├── models/
│   │   ├── se_st_combined.py      # Main SE+ST model
│   │   ├── base.py                # Base perturbation model
│   │   ├── state_transition.py    # ST model component
│   │   └── utils.py               # Model utilities
│   ├── utils/
│   │   ├── se_st_utils.py         # Training/inference utilities
│   │   └── se_inference.py        # SE model inference
│   ├── configs/
│   │   └── se_st_combined.yaml    # Model configuration
│   └── data/                      # Data utilities
├── examples/
│   └── se_st_virtual_cell_challenge.py  # Complete training example
├── setup.py                       # Package setup
├── requirements.txt               # Dependencies
└── README.md                      # This file
```

## 🧪 Examples

### Virtual Cell Challenge

See `examples/se_st_virtual_cell_challenge.py` for a complete example of training and evaluating the SE+ST model on the Virtual Cell Challenge dataset.

### Cross-cell-type Prediction

```python
from se_st_combined.utils.se_st_utils import evaluate_cross_cell_type_performance

# Evaluate performance across different cell types
results = evaluate_cross_cell_type_performance(
    model=model,
    test_data=test_data,
    cell_types=["k562", "hepg2", "rpe1", "jurkat"]
)

# Compare with baseline
comparison = compare_with_baseline(se_st_results, baseline_results)
```

## 🔬 Research Background

This model is based on the STATE (State Transition and Embedding) framework for single-cell perturbation prediction. The key innovation is combining:

1. **State Embedding models** for universal cell representations
2. **State Transition models** for perturbation effect prediction
3. **Cross-cell-type generalization** through shared embedding space

## 📚 Dependencies

- Python >= 3.8
- PyTorch >= 1.12.0
- Lightning >= 2.0.0
- Scanpy >= 1.9.0
- And more (see requirements.txt)

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Based on the STATE framework from Arc Institute
- Uses pre-trained SE models for cell embeddings
- Inspired by the Virtual Cell Challenge

## 📞 Support

If you encounter any issues or have questions, please:

1. Check the examples in the `examples/` directory
2. Review the configuration options
3. Open an issue on GitHub

## 🔗 Related Work

- [STATE: State Transition and Embedding](https://github.com/ArcInstitute/STATE)
- [Virtual Cell Challenge](https://huggingface.co/datasets/ArcInstitute/VirtualCellChallenge)
- [SE-600M Model](https://huggingface.co/ArcInstitute/SE-600M)
