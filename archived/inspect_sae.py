"""
Quick script to download and inspect the SAE checkpoint structure.
Run this locally (CPU only, needs ~5GB disk for download) to verify
the weight format before renting GPU.
"""

import torch
from huggingface_hub import hf_hub_download

print("Downloading SAE checkpoint...")
path = hf_hub_download(
    repo_id="Goodfire/Llama-3.3-70B-Instruct-SAE-l50",
    filename="Llama-3.3-70B-Instruct-SAE-l50.pt",
    repo_type="model",
)
print(f"Downloaded to: {path}")

print("\nLoading checkpoint (CPU)...")
state_dict = torch.load(path, map_location="cpu", weights_only=True)

print(f"\nCheckpoint type: {type(state_dict)}")

if isinstance(state_dict, dict):
    print(f"Number of keys: {len(state_dict)}")
    print("\nKeys and shapes:")
    for key, val in state_dict.items():
        if isinstance(val, torch.Tensor):
            print(f"  {key}: shape={val.shape}, dtype={val.dtype}, "
                  f"size={val.numel() * val.element_size() / 1e9:.2f} GB")
        elif isinstance(val, dict):
            print(f"  {key}: dict with {len(val)} keys")
            for k2, v2 in list(val.items())[:5]:
                if isinstance(v2, torch.Tensor):
                    print(f"    {k2}: shape={v2.shape}, dtype={v2.dtype}")
                else:
                    print(f"    {k2}: {type(v2)}")
            if len(val) > 5:
                print(f"    ... ({len(val) - 5} more)")
        else:
            print(f"  {key}: {type(val)} = {val}")
elif isinstance(state_dict, torch.Tensor):
    print(f"Single tensor: shape={state_dict.shape}, dtype={state_dict.dtype}")
else:
    print(f"Unexpected type: {type(state_dict)}")
    print(f"Content: {state_dict}")

# Try to infer SAE dimensions
print("\n" + "=" * 60)
print("INFERRED SAE ARCHITECTURE:")
print("=" * 60)

if isinstance(state_dict, dict):
    # Find weight matrices
    for key, val in state_dict.items():
        if isinstance(val, torch.Tensor) and val.dim() == 2:
            d0, d1 = val.shape
            if d0 < d1:
                print(f"  {key}: likely encoder (d_in={d0}, d_hidden={d1}, expansion={d1/d0:.1f}x)")
            else:
                print(f"  {key}: likely decoder or transposed encoder (d0={d0}, d1={d1})")

print("\nExpected for Llama 3.3 70B:")
print("  d_in = 8192 (hidden dim)")
print("  d_hidden = ? (expansion * 8192)")
print("  L0 = 121 (from model card)")
