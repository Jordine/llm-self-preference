"""
Sparse Autoencoder for Goodfire Llama 3.3 70B SAE weights.

Checkpoint format (from inspection):
  encoder_linear.weight: (65536, 8192) — maps d_in → d_hidden
  encoder_linear.bias:   (65536,)
  decoder_linear.weight: (8192, 65536) — maps d_hidden → d_in
  decoder_linear.bias:   (8192,)

Architecture: d_in=8192, d_hidden=65536, expansion=8x, L0=121
Untied weights (separate encoder and decoder).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from typing import Optional


class GoodfireSAE(nn.Module):
    """
    Sparse autoencoder with top-k sparsity and untied encoder/decoder weights.

    Matches Goodfire's Llama-3.3-70B-Instruct-SAE-l50 checkpoint format:
    - encoder_linear: (d_hidden, d_in) weight + (d_hidden,) bias
    - decoder_linear: (d_in, d_hidden) weight + (d_in,) bias
    """

    def __init__(self, d_in: int, d_hidden: int, k: int, dtype: torch.dtype = torch.bfloat16):
        super().__init__()
        self.d_in = d_in
        self.d_hidden = d_hidden
        self.k = k
        self.dtype = dtype

        self.encoder_linear = nn.Linear(d_in, d_hidden, dtype=dtype)
        self.decoder_linear = nn.Linear(d_hidden, d_in, dtype=dtype)

    def encode_pre(self, x: torch.Tensor) -> torch.Tensor:
        """Linear encoder transform before activation."""
        return self.encoder_linear(x)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input to sparse feature activations (ReLU + top-k)."""
        pre = self.encode_pre(x)
        pre = F.relu(pre)
        return self._topk(pre)

    def _topk(self, x: torch.Tensor) -> torch.Tensor:
        """Keep only top-k activations per token, zero out the rest."""
        if x.dim() <= 2:
            _, topk_indices = torch.topk(x, self.k, dim=-1)
            mask = torch.zeros_like(x, dtype=torch.bool)
            mask.scatter_(-1, topk_indices, True)
        elif x.dim() == 3:
            # (batch, seq_len, d_hidden)
            orig_shape = x.shape
            flat = x.reshape(-1, x.shape[-1])
            _, topk_indices = torch.topk(flat, self.k, dim=-1)
            mask = torch.zeros_like(flat, dtype=torch.bool)
            mask.scatter_(-1, topk_indices, True)
            mask = mask.view(orig_shape)
        else:
            raise ValueError(f"Unexpected input dimension: {x.dim()}")
        return x * mask

    def decode(self, features: torch.Tensor) -> torch.Tensor:
        """Decode sparse features back to input space."""
        return self.decoder_linear(features)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Full forward pass: encode then decode.
        Returns (reconstructed, features).
        """
        features = self.encode(x)
        reconstructed = self.decode(features)
        return reconstructed, features


def load_sae_from_checkpoint(
    path: str,
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
    k: int = 121,
) -> GoodfireSAE:
    """
    Load the Goodfire SAE from a .pt checkpoint.
    """
    state_dict = torch.load(path, map_location="cpu", weights_only=True)

    print("SAE checkpoint keys:")
    for key, val in state_dict.items():
        if isinstance(val, torch.Tensor):
            print(f"  {key}: shape={val.shape}, dtype={val.dtype}")

    # Infer dimensions from encoder weight
    enc_w = state_dict["encoder_linear.weight"]  # (d_hidden, d_in)
    d_hidden, d_in = enc_w.shape
    print(f"\n  d_in={d_in}, d_hidden={d_hidden}, expansion={d_hidden/d_in:.1f}x, k={k}")

    sae = GoodfireSAE(d_in=d_in, d_hidden=d_hidden, k=k, dtype=dtype)

    # Convert checkpoint to target dtype and load
    converted = {key: v.to(dtype) for key, v in state_dict.items()}
    sae.load_state_dict(converted, strict=True)

    sae = sae.to(device=device)
    sae.eval()
    print(f"  SAE loaded on {device}, dtype={dtype}")
    return sae


def download_and_load_llama_sae(
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
    k: int = 121,
    cache_dir: Optional[str] = None,
) -> GoodfireSAE:
    """
    Download the Goodfire Llama 3.3 70B SAE from HuggingFace and load it.
    Targets layer 50 of meta-llama/Llama-3.3-70B-Instruct.
    """
    print("Downloading SAE weights from HuggingFace...")
    path = hf_hub_download(
        repo_id="Goodfire/Llama-3.3-70B-Instruct-SAE-l50",
        filename="Llama-3.3-70B-Instruct-SAE-l50.pt",
        repo_type="model",
        cache_dir=cache_dir,
    )
    print(f"  Downloaded to: {path}")
    return load_sae_from_checkpoint(path, device=device, dtype=dtype, k=k)


if __name__ == "__main__":
    import sys

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if len(sys.argv) > 1:
        sae = load_sae_from_checkpoint(sys.argv[1], device=device)
    else:
        sae = download_and_load_llama_sae(device=device)

    print(f"\nSAE summary:")
    print(f"  Parameters: {sum(p.numel() for p in sae.parameters()):,}")
    print(f"  d_in={sae.d_in}, d_hidden={sae.d_hidden}, k={sae.k}")

    # Test with random input
    x = torch.randn(1, sae.d_in, dtype=sae.dtype, device=device)
    reconstructed, features = sae(x)
    print(f"\n  Input: {x.shape}")
    print(f"  Features: {features.shape}, active: {(features > 0).sum().item()}")
    print(f"  Reconstruction MSE: {F.mse_loss(x, reconstructed).item():.6f}")
