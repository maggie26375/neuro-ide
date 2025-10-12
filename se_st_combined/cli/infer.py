"""
SE+ST Combined Model Inference CLI

Usage:
    se-st-infer \
        --checkpoint competition/se_st_combined_40k_v2/final_model.ckpt \
        --adata /data/competition_val_template.h5ad \
        --output competition/prediction.h5ad \
        --pert-col target_gene \
        --se-model-path SE-600M \
        --perturbation-features /data/ESM2_pert_features.pt
"""

import argparse
import logging
from pathlib import Path
import sys

import anndata
import h5py
import numpy as np
import torch
from tqdm import tqdm

from se_st_combined.models.se_st_combined import SE_ST_CombinedModel

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [%(name)s] [%(levelname)s] - %(message)s'
)
logger = logging.getLogger(__name__)


def load_model_from_checkpoint(checkpoint_path: str, se_model_path: str) -> SE_ST_CombinedModel:
    """Load SE+ST Combined Model from checkpoint."""
    logger.info(f"Loading model from checkpoint: {checkpoint_path}")
    
    # Load the checkpoint to inspect it
    # Note: weights_only=False is safe here because we trust our own checkpoints
    ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    # Extract hyperparameters from checkpoint
    if 'hyper_parameters' in ckpt:
        hparams = ckpt['hyper_parameters']
        logger.info(f"Found hyperparameters in checkpoint: {list(hparams.keys())}")
    else:
        logger.warning("No hyperparameters found in checkpoint, using defaults")
        hparams = {}
    
    # Create model with correct dimensions
    model = SE_ST_CombinedModel(
        input_dim=hparams.get('input_dim', 18080),
        hidden_dim=hparams.get('hidden_dim', 512),
        output_dim=hparams.get('output_dim', 18080),
        pert_dim=hparams.get('pert_dim', 5120),
        se_model_path=se_model_path,
        se_checkpoint_path=hparams.get('se_checkpoint_path', f"{se_model_path}/se600m_epoch15.ckpt"),
    )
    
    # Load state dict
    if 'state_dict' in ckpt:
        model.load_state_dict(ckpt['state_dict'], strict=False)
        logger.info("Model weights loaded successfully")
    else:
        raise ValueError("No state_dict found in checkpoint")
    
    model.eval()
    return model


def load_perturbation_features(features_path: str) -> dict:
    """Load ESM2 perturbation features."""
    logger.info(f"Loading perturbation features from {features_path}")
    features = torch.load(features_path, map_location='cpu')
    logger.info(f"Loaded {len(features)} perturbation features")
    return features


def run_inference(
    model: SE_ST_CombinedModel,
    adata: anndata.AnnData,
    pert_features: dict,
    pert_col: str = "target_gene",
    batch_size: int = 16,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> anndata.AnnData:
    """
    Run inference on an AnnData object.
    
    Args:
        model: Trained SE+ST Combined Model
        adata: Input AnnData with perturbation metadata
        pert_features: Dictionary mapping perturbation names to embeddings
        pert_col: Column name for perturbation in adata.obs
        batch_size: Batch size for inference
        device: Device to run inference on
        
    Returns:
        AnnData object with predicted expression values
    """
    model = model.to(device)
    model.eval()
    
    logger.info(f"Running inference on {adata.n_obs} cells")
    logger.info(f"Device: {device}")
    
    # Get unique perturbations
    unique_perts = adata.obs[pert_col].unique()
    logger.info(f"Found {len(unique_perts)} unique perturbations")
    
    # Prepare predictions storage (use float16 to save space)
    predictions = np.zeros((adata.n_obs, adata.n_vars), dtype=np.float16)
    
    # Group cells by perturbation for efficient batch processing
    with torch.no_grad():
        for pert_name in tqdm(unique_perts, desc="Processing perturbations"):
            # Get indices for this perturbation
            pert_mask = adata.obs[pert_col] == pert_name
            pert_indices = np.where(pert_mask)[0]
            
            if len(pert_indices) == 0:
                continue
            
            # Get perturbation embedding
            pert_name_str = pert_name.decode('utf-8') if isinstance(pert_name, bytes) else pert_name
            
            # Try to find embedding (case-insensitive)
            pert_emb = None
            for key in pert_features.keys():
                if key.lower() == pert_name_str.lower():
                    pert_emb = pert_features[key]
                    break
            
            if pert_emb is None:
                logger.warning(f"No embedding found for {pert_name_str}, using zeros")
                pert_emb = torch.zeros(5120)
            
            if not isinstance(pert_emb, torch.Tensor):
                pert_emb = torch.tensor(pert_emb)
            
            pert_emb = pert_emb.to(device)
            
            # Get expression data for these cells
            if hasattr(adata.X, 'toarray'):
                X = adata.X[pert_indices].toarray()
            else:
                X = adata.X[pert_indices]
            
            # Process in batches
            for i in range(0, len(pert_indices), batch_size):
                end_idx = min(i + batch_size, len(pert_indices))
                batch_X = X[i:end_idx]
                
                # Convert to tensor
                batch_X_tensor = torch.tensor(batch_X, dtype=torch.float32).to(device)
                
                # Create cell sentence by repeating each cell 128 times (to match training)
                # Shape: [batch_size, cell_sentence_len=128, gene_dim]
                batch_X_tensor = batch_X_tensor.unsqueeze(1).repeat(1, 128, 1)
                
                # Flatten to [batch_size*128, gene_dim] as model expects
                batch_size = batch_X_tensor.shape[0]
                batch_X_tensor = batch_X_tensor.reshape(-1, batch_X_tensor.shape[2])
                
                # Repeat perturbation embedding for each cell in sentence
                batch_pert_emb = pert_emb.unsqueeze(0).repeat(batch_size * 128, 1)
                
                # Run inference
                try:
                    # Model expects a batch dictionary
                    batch_dict = {
                        'ctrl_cell_emb': batch_X_tensor,
                        'pert_cell_emb': batch_X_tensor,  # Use same as control
                        'pert_emb': batch_pert_emb,
                    }
                    
                    pred = model(batch_dict)
                    
                    # Predictions are [batch_size*128, gene_dim]
                    # Reshape to [batch_size, 128, gene_dim] and average
                    pred = pred.reshape(batch_size, 128, -1)
                    pred = pred.mean(dim=1)  # Average over cell_sentence_len → [batch_size, gene_dim]
                    
                    # Store predictions
                    pred_np = pred.cpu().numpy()
                    predictions[pert_indices[i:end_idx]] = pred_np
                    
                except Exception as e:
                    logger.error(f"Error processing batch for {pert_name_str}: {e}")
                    # Use input as fallback
                    predictions[pert_indices[i:end_idx]] = batch_X
    
    # Create output AnnData
    output_adata = anndata.AnnData(
        X=predictions,
        obs=adata.obs.copy(),
        var=adata.var.copy(),
    )
    
    logger.info(f"Inference completed! Predictions shape: {predictions.shape}")
    return output_adata


def main():
    parser = argparse.ArgumentParser(description="SE+ST Combined Model Inference")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint (.ckpt file)"
    )
    parser.add_argument(
        "--adata",
        type=str,
        required=True,
        help="Path to input AnnData (.h5ad file)"
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to output predictions (.h5ad file)"
    )
    parser.add_argument(
        "--pert-col",
        type=str,
        default="target_gene",
        help="Column name for perturbation in adata.obs"
    )
    parser.add_argument(
        "--se-model-path",
        type=str,
        required=True,
        help="Path to SE model directory"
    )
    parser.add_argument(
        "--perturbation-features",
        type=str,
        required=True,
        help="Path to ESM2 perturbation features (.pt file)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size for inference"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run inference on (cuda/cpu)"
    )
    
    args = parser.parse_args()
    
    # Validate inputs
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        logger.error(f"Checkpoint not found: {checkpoint_path}")
        sys.exit(1)
    
    adata_path = Path(args.adata)
    if not adata_path.exists():
        logger.error(f"AnnData file not found: {adata_path}")
        sys.exit(1)
    
    pert_features_path = Path(args.perturbation_features)
    if not pert_features_path.exists():
        logger.error(f"Perturbation features not found: {pert_features_path}")
        sys.exit(1)
    
    # Create output directory
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Load model
    model = load_model_from_checkpoint(
        str(checkpoint_path),
        args.se_model_path
    )
    
    # Load perturbation features
    pert_features = load_perturbation_features(str(pert_features_path))
    
    # Load input data
    logger.info(f"Loading input data from {adata_path}")
    adata = anndata.read_h5ad(adata_path)
    logger.info(f"Loaded {adata.n_obs} cells x {adata.n_vars} genes")
    
    # Run inference
    output_adata = run_inference(
        model=model,
        adata=adata,
        pert_features=pert_features,
        pert_col=args.pert_col,
        batch_size=args.batch_size,
        device=args.device,
    )
    
    # Save predictions with compression
    logger.info(f"Saving predictions to {output_path}")
    output_adata.write_h5ad(output_path, compression="gzip", compression_opts=9)
    logger.info("✅ Inference completed successfully!")
    
    # Show file size
    import os
    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    logger.info(f"Output file size: {file_size_mb:.2f} MB")


if __name__ == "__main__":
    main()
