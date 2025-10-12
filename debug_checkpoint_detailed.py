"""
Detailed checkpoint inspection to find dimension mismatch
"""
import torch

checkpoint_path = "/Users/maggie/Downloads/final_model.ckpt"  # 你下载的本地路径
ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

print("=" * 80)
print("📊 DETAILED CHECKPOINT INSPECTION")
print("=" * 80)

# Hyperparameters
if 'hyper_parameters' in ckpt:
    hparams = ckpt['hyper_parameters']
    print("\n🔍 Hyperparameters:")
    for key, value in hparams.items():
        print(f"  {key}: {value}")

# Find all encoder/decoder layer shapes
print("\n" + "=" * 80)
print("🔍 ST MODEL LAYER DIMENSIONS")
print("=" * 80)

state_dict = ckpt['state_dict']

# pert_encoder
print("\n📌 pert_encoder layers:")
for key in sorted(state_dict.keys()):
    if 'st_model.pert_encoder' in key and 'weight' in key:
        shape = state_dict[key].shape
        print(f"  {key}: {shape}")
        if len(shape) == 2:
            print(f"    → Input: {shape[1]}, Output: {shape[0]}")

# basal_encoder  
print("\n📌 basal_encoder layers:")
for key in sorted(state_dict.keys()):
    if 'st_model.basal_encoder' in key and 'weight' in key:
        shape = state_dict[key].shape
        print(f"  {key}: {shape}")
        if len(shape) == 2:
            print(f"    → Input: {shape[1]}, Output: {shape[0]}")

# project_out
print("\n📌 project_out layers:")
for key in sorted(state_dict.keys()):
    if 'st_model.project_out' in key and 'weight' in key:
        shape = state_dict[key].shape
        print(f"  {key}: {shape}")
        if len(shape) == 2:
            print(f"    → Input: {shape[1]}, Output: {shape[0]}")

# SE model (if exists)
print("\n📌 SE model layers:")
se_found = False
for key in sorted(state_dict.keys()):
    if 'se_model' in key or 'se_inference' in key:
        se_found = True
        if 'weight' in key:
            shape = state_dict[key].shape
            print(f"  {key}: {shape}")
            if len(shape) == 2:
                print(f"    → Input: {shape[1]}, Output: {shape[0]}")

if not se_found:
    print("  ⚠️  No SE model found in checkpoint!")

# Key dimensions summary
print("\n" + "=" * 80)
print("🎯 KEY DIMENSIONS SUMMARY")
print("=" * 80)

# Find first layer of each encoder to determine input dims
pert_encoder_first = state_dict.get('st_model.pert_encoder.0.weight')
basal_encoder_first = state_dict.get('st_model.basal_encoder.0.weight')
project_out_last = state_dict.get('st_model.project_out.9.weight')

if pert_encoder_first is not None:
    print(f"\n✅ pert_encoder input_dim: {pert_encoder_first.shape[1]}")
    print(f"✅ pert_encoder output_dim (hidden): {pert_encoder_first.shape[0]}")

if basal_encoder_first is not None:
    print(f"\n✅ basal_encoder input_dim: {basal_encoder_first.shape[1]}")
    print(f"   → This is what ST model expects for ctrl_cell_emb!")
    print(f"✅ basal_encoder output_dim (hidden): {basal_encoder_first.shape[0]}")

if project_out_last is not None:
    print(f"\n✅ project_out final output_dim: {project_out_last.shape[0]}")
    print(f"   → This is the predicted gene expression dimension!")

print("\n" + "=" * 80)
print("⚠️  CRITICAL CHECKS")
print("=" * 80)

if basal_encoder_first is not None:
    expected_ctrl_dim = basal_encoder_first.shape[1]
    print(f"\n1. ctrl_cell_emb must have dimension: {expected_ctrl_dim}")
    print(f"   Current SE output_dim: {hparams.get('hidden_dim', 'Unknown')}")
    
    if hparams.get('hidden_dim') != expected_ctrl_dim:
        print(f"   ❌ MISMATCH! SE outputs {hparams.get('hidden_dim')}, but ST expects {expected_ctrl_dim}")
    else:
        print(f"   ✅ Dimensions match!")

print("\n" + "=" * 80)

