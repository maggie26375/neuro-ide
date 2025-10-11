# Architecture Fix: Proper Cell Sentences

## Problem
Current implementation returns single cell pairs, incompatible with STATE's cell_sentence_len=128 design.

## STATE's Architecture
- **Cell Sentence**: A sequence of cells (default 128)
- **Purpose**: 
  - Learn robust patterns across multiple cells
  - Reduce single-cell noise
  - Leverage Transformer's attention mechanism

## Required Data Format
For each training sample:
```python
{
    'ctrl_cell_emb': torch.Tensor,  # [cell_sentence_len, gene_dim]  e.g., [128, 18080]
    'pert_cell_emb': torch.Tensor,  # [cell_sentence_len, gene_dim]  e.g., [128, 18080]
    'pert_emb': torch.Tensor,       # [cell_sentence_len, pert_dim]  e.g., [128, 5120]
}
```

## Solution
Modify `_load_and_pair_single_h5` to:
1. For each perturbation, sample `cell_sentence_len` perturbed cells
2. Sample `cell_sentence_len` control cells (same batch/cell type)
3. Create ONE sample with ALL these cells as sequences
4. Repeat perturbation embedding for each cell in the sentence

This matches STATE's original design and will give MUCH better performance!

