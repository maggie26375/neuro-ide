# Neural ODE Analysis and Visualization Tools

This module provides comprehensive analysis and visualization tools for understanding the dynamics of Neural ODE perturbation models.

## Features

### 1. `analyze_perturbation_dynamics()`

Analyzes the dynamics of perturbation responses in the Neural ODE model.

**Outputs:**
- Full state trajectory over time
- Velocity field at each time point
- Perturbation response magnitude
- State statistics (mean, std, min, max)
- Convergence metrics
- Intrinsic dimensionality

**Example:**
```python
from models.neural_ode_perturbation import NeuralODEPerturbationModel
from utils.neural_ode_analysis import analyze_perturbation_dynamics

# Create model
model = NeuralODEPerturbationModel(
    state_dim=64,
    pert_dim=128,
    gene_dim=100
)

# Prepare data
initial_states = torch.randn(8, 64)
perturbation_emb = torch.randn(8, 128)

# Run analysis
results = analyze_perturbation_dynamics(
    model,
    initial_states,
    perturbation_emb,
    num_time_points=50
)

# Access results
print(f"Trajectory shape: {results['trajectory'].shape}")
print(f"Convergence metrics: {results['convergence_metrics']}")
```

### 2. `visualize_perturbation_dynamics()`

Creates a comprehensive multi-panel visualization of perturbation dynamics.

**Visualization Panels:**
1. **Response Magnitude Over Time**: Shows how far states move from initial conditions
2. **Velocity Magnitude Over Time**: Shows the speed of state changes
3. **Convergence Metrics**: Text summary of convergence properties
4. **State Statistics**: Evolution of mean state values
5. **State Variance**: Variance across the batch over time
6. **PCA Projection**: 2D trajectory visualization in principal component space
7. **3D Trajectory**: Individual trajectories in 3D state space
8. **Velocity Heatmap**: Time-series heatmap of velocity magnitudes
9. **Phase Portrait**: State vs. velocity showing system dynamics

**Example:**
```python
from utils.neural_ode_analysis import visualize_perturbation_dynamics

# Visualize the results
fig = visualize_perturbation_dynamics(
    results,
    save_path='outputs/dynamics_analysis.png',
    sample_trajectories=5,
    pca_components=2,
    show_velocity_field=True,
    figsize=(20, 12),
    dpi=100
)
```

### 3. `compare_perturbations()`

Compares dynamics across multiple different perturbations.

**Comparison Views:**
- Response magnitude comparison
- Velocity comparison
- Convergence metrics bar chart
- Final state distribution histograms

**Example:**
```python
from utils.neural_ode_analysis import compare_perturbations

# Create different perturbations
pert1 = torch.randn(8, 128) * 0.5  # Small
pert2 = torch.randn(8, 128) * 1.0  # Medium
pert3 = torch.randn(8, 128) * 1.5  # Large

perturbations = [pert1, pert2, pert3]
names = ['Small', 'Medium', 'Large']

# Compare
fig = compare_perturbations(
    model,
    initial_states,
    perturbations,
    perturbation_names=names,
    save_path='outputs/perturbation_comparison.png'
)
```

### 4. `export_analysis_data()`

Exports analysis results to files for further processing.

**Exported Files:**
- `*_trajectory.npy`: Full trajectory array
- `*_velocity_field.npy`: Velocity field array
- `*_time_points.npy`: Time points used
- `*_metrics.txt`: Text summary of convergence metrics

**Example:**
```python
from utils.neural_ode_analysis import export_analysis_data

export_analysis_data(
    results,
    output_dir='analysis_outputs',
    prefix='experiment_1'
)
```

## Understanding the Metrics

### Convergence Metrics

- **Final Velocity Mean**: Average velocity in the final 20% of trajectory
  - Lower values indicate the system is settling down

- **Final Velocity Std**: Standard deviation of final velocities
  - Lower values indicate consistent behavior across samples

- **Trajectory Length**: Total distance traveled from initial state
  - Indicates magnitude of perturbation response

- **Is Converging**: Boolean indicating if velocity is decreasing
  - `True` if final velocity < average velocity

- **Convergence Rate**: Ratio of initial to final velocity
  - Higher values (>1.0) indicate faster convergence
  - Lower values (<1.0) indicate diverging or oscillating behavior

- **Intrinsic Dimensionality**: Effective number of dimensions used
  - Based on PCA eigenvalue analysis
  - Lower values indicate simpler dynamics

## Interpreting Visualizations

### Response Magnitude Plot
- **Increasing trend**: System moving away from initial state
- **Plateauing**: System reaching steady state
- **Large variance (shaded area)**: Heterogeneous responses across batch

### Velocity Plot
- **Decreasing**: System converging to equilibrium
- **Constant**: Uniform state evolution
- **Increasing**: Potential instability or divergence

### PCA Projection
- **Smooth curve**: Orderly progression through state space
- **Loops/spirals**: Oscillatory dynamics
- **Linear path**: Monotonic state change
- **Color gradient**: Shows temporal progression (blue=early, yellow=late)

### Phase Portrait
- **Arrows pointing inward**: Stable fixed point
- **Arrows pointing outward**: Unstable fixed point
- **Circular patterns**: Limit cycle (oscillation)
- **Complex patterns**: Chaotic or multi-stable dynamics

## Testing

Run the comprehensive test suite:

```bash
python test_neural_ode_analysis.py
```

This will:
1. Test trajectory analysis
2. Create visualization examples
3. Compare multiple perturbations
4. Export analysis data
5. Verify trajectory consistency

All test outputs will be saved to `test_outputs/` directory.

## Dependencies

Core dependencies (automatically installed with neuro-IDE):
- `torch>=1.12.0`
- `numpy>=1.21.0`
- `matplotlib>=3.5.0`
- `seaborn>=0.11.0`
- `scipy>=1.7.0`
- `scikit-learn>=1.0.0`
- `torchdiffeq>=0.2.3`

## Tips for Analysis

1. **Start with fewer time points** (10-20) for quick exploration, then increase for detailed analysis
2. **Use sample_trajectories=3-5** to avoid cluttered plots
3. **Compare perturbations of different magnitudes** to understand dose-response relationships
4. **Check convergence metrics** to ensure ODE solver is producing stable solutions
5. **Use PCA projection** to visualize high-dimensional trajectories
6. **Examine phase portraits** to understand attractor structure

## Common Issues

### ODE Solver Warnings
If you see warnings about solver tolerances:
- Increase `rtol` and `atol` in the model's ODE solver
- Try different solver methods ('euler', 'rk4', 'dopri5')

### Memory Issues
For large batches or many time points:
- Reduce `batch_size`
- Reduce `num_time_points`
- Use `device='cpu'` instead of GPU

### Visualization Issues
If plots are too cluttered:
- Reduce `sample_trajectories`
- Increase `figsize`
- Save to file instead of displaying

## Citation

If you use these analysis tools in your research, please cite:

```bibtex
@software{neuro_ide_analysis,
  title={Neural ODE Analysis and Visualization Tools},
  author={neuro-IDE Development Team},
  year={2024},
  url={https://github.com/maggie26375/neuro-ide}
}
```
