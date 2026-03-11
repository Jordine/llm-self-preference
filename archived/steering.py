"""
SAE-based steering for Llama 3.3 70B.

Hooks into the model's forward pass at layer 50, runs SAE encode/decode
with feature modifications to steer model behavior.
"""

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Optional
from dataclasses import dataclass, field
from sae import GoodfireSAE, download_and_load_llama_sae


@dataclass
class FeatureIntervention:
    """A single feature steering intervention."""
    feature_idx: int
    magnitude: float  # positive = amplify, negative = suppress
    label: str = ""  # human-readable label for logging


@dataclass
class SteeringConfig:
    """Configuration for a steering experiment."""
    interventions: list[FeatureIntervention] = field(default_factory=list)
    target_layer: int = 50  # Layer to hook into (0-indexed)
    method: str = "add"  # "add" (additive), "set" (override), "scale" (multiplicative)

    def describe(self) -> str:
        parts = []
        for iv in self.interventions:
            name = iv.label or f"feat_{iv.feature_idx}"
            parts.append(f"{name}={iv.magnitude:+.2f}")
        return ", ".join(parts) if parts else "no interventions"


class SAESteeringHook:
    """
    Hooks into a transformer layer to apply SAE-based feature steering.

    During the forward pass:
    1. Intercepts residual stream activations after the target layer
    2. Encodes through SAE to get feature activations
    3. Modifies specified features (add/set/scale)
    4. Decodes back and replaces the residual stream
    """

    def __init__(
        self,
        sae: GoodfireSAE,
        config: SteeringConfig,
        device: str = "cuda",
    ):
        self.sae = sae
        self.config = config
        self.device = device
        self._hook_handle = None
        self._last_features = None  # Store for inspection

    def _hook_fn(self, module, input, output):
        """
        Hook function that intercepts layer output and applies SAE steering.

        For Llama models, each decoder layer outputs a tuple:
        (hidden_states, ...) where hidden_states is (batch, seq_len, d_model)
        """
        # Extract hidden states from layer output
        if isinstance(output, tuple):
            hidden_states = output[0]
            rest = output[1:]
        else:
            hidden_states = output
            rest = None

        # Process each position in the sequence
        original_shape = hidden_states.shape
        original_dtype = hidden_states.dtype
        flat = hidden_states.to(self.sae.dtype)

        # Encode through SAE
        features = self.sae.encode(flat)
        self._last_features = features.detach()

        # Apply interventions
        if self.config.interventions:
            features = self._apply_interventions(features)

        # Decode back
        reconstructed = self.sae.decode(features)

        # Compute the steering delta and add it to original hidden states
        # This preserves information not captured by the SAE
        # delta = reconstructed - sae.decode(sae.encode(original))
        # steered = original + delta
        # Equivalently: steered = original + (modified_recon - original_recon)
        original_recon = self.sae.decode(self.sae.encode(flat))
        delta = reconstructed - original_recon
        steered = flat + delta

        steered = steered.to(original_dtype)

        if rest is not None:
            return (steered,) + rest
        return steered

    def _apply_interventions(self, features: torch.Tensor) -> torch.Tensor:
        """Apply feature interventions to the encoded features."""
        features = features.clone()

        for iv in self.config.interventions:
            idx = iv.feature_idx
            if self.config.method == "add":
                features[..., idx] = features[..., idx] + iv.magnitude
            elif self.config.method == "set":
                features[..., idx] = iv.magnitude
            elif self.config.method == "scale":
                features[..., idx] = features[..., idx] * iv.magnitude
            else:
                raise ValueError(f"Unknown method: {self.config.method}")

            # Ensure non-negative (SAE features are post-ReLU)
            if self.config.method != "set":
                features[..., idx] = F.relu(features[..., idx])

        return features

    def register(self, model) -> None:
        """Register the hook on the target layer of the model."""
        self.remove()
        target_layer = model.model.layers[self.config.target_layer]
        self._hook_handle = target_layer.register_forward_hook(self._hook_fn)
        print(f"Hook registered on layer {self.config.target_layer}")

    def remove(self) -> None:
        """Remove the hook."""
        if self._hook_handle is not None:
            self._hook_handle.remove()
            self._hook_handle = None

    def get_last_features(self) -> Optional[torch.Tensor]:
        """Get the feature activations from the last forward pass."""
        return self._last_features


class SteeredLlama:
    """
    Wrapper around Llama 3.3 70B with SAE-based steering capabilities.

    Usage:
        model = SteeredLlama.load()
        model.set_steering([FeatureIntervention(feature_idx=42, magnitude=0.5)])
        response = model.generate("Are you conscious?")
        model.clear_steering()
    """

    def __init__(
        self,
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        sae: GoodfireSAE,
        device: str = "cuda",
        target_layer: int = 50,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.sae = sae
        self.device = device
        self.target_layer = target_layer
        self._hook: Optional[SAESteeringHook] = None

    @classmethod
    def load(
        cls,
        model_name: str = "meta-llama/Llama-3.3-70B-Instruct",
        device_map: str = "auto",
        dtype: torch.dtype = torch.bfloat16,
        sae_k: int = 121,
        target_layer: int = 50,
        sae_cache_dir: Optional[str] = None,
        attn_implementation: str = "flash_attention_2",
    ) -> "SteeredLlama":
        """Load model, tokenizer, and SAE."""
        print(f"Loading tokenizer: {model_name}")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        print(f"Loading model: {model_name}")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            device_map=device_map,
            attn_implementation=attn_implementation,
        )
        model.eval()

        # Determine which device the target layer is on
        target_layer_module = model.model.layers[target_layer]
        layer_device = next(target_layer_module.parameters()).device
        print(f"Target layer {target_layer} is on device: {layer_device}")

        print("Loading SAE...")
        sae = download_and_load_llama_sae(
            device=str(layer_device),
            dtype=dtype,
            k=sae_k,
            cache_dir=sae_cache_dir,
        )

        return cls(
            model=model,
            tokenizer=tokenizer,
            sae=sae,
            device=str(layer_device),
            target_layer=target_layer,
        )

    def set_steering(
        self,
        interventions: list[FeatureIntervention],
        method: str = "add",
    ) -> None:
        """Set feature steering interventions."""
        self.clear_steering()
        config = SteeringConfig(
            interventions=interventions,
            target_layer=self.target_layer,
            method=method,
        )
        self._hook = SAESteeringHook(self.sae, config, self.device)
        self._hook.register(self.model)
        print(f"Steering active: {config.describe()}")

    def clear_steering(self) -> None:
        """Remove all steering interventions."""
        if self._hook is not None:
            self._hook.remove()
            self._hook = None

    def generate(
        self,
        messages: list[dict[str, str]],
        max_new_tokens: int = 256,
        temperature: float = 0.0,
        do_sample: bool = False,
        seed: Optional[int] = None,
    ) -> str:
        """Generate a response with optional steering."""
        # Format as chat
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)

        if seed is not None:
            torch.manual_seed(seed)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature if do_sample else None,
                do_sample=do_sample,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        # Decode only the new tokens
        new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        response = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
        return response

    def get_feature_activations(
        self,
        messages: list[dict[str, str]],
    ) -> torch.Tensor:
        """
        Run a forward pass and return SAE feature activations at the target layer.
        Useful for feature search/discovery.
        """
        # Ensure we have a hook to capture activations (with no interventions)
        was_steering = self._hook is not None
        if not was_steering:
            self.set_steering([])  # Empty interventions, just to capture activations

        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            self.model(**inputs)

        features = self._hook.get_last_features()

        if not was_steering:
            self.clear_steering()

        return features

    def get_token_features(
        self,
        messages: list[dict[str, str]],
    ) -> tuple[list[str], torch.Tensor]:
        """
        Get per-token feature activations. Returns (tokens, features).
        features shape: (seq_len, d_hidden)
        """
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        tokens = self.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

        features = self.get_feature_activations(messages)
        # features is (1, seq_len, d_hidden) — squeeze batch dim
        if features.dim() == 3:
            features = features[0]

        return tokens, features


if __name__ == "__main__":
    # Quick integration test
    print("Loading SteeredLlama...")
    llama = SteeredLlama.load()

    # Test without steering
    messages = [
        {"role": "user", "content": "What is 2+2?"}
    ]
    print("\n--- No steering ---")
    response = llama.generate(messages, max_new_tokens=50)
    print(f"Response: {response}")

    # Test with a random feature intervention
    print("\n--- With steering (feature 100, magnitude +0.5) ---")
    llama.set_steering([FeatureIntervention(feature_idx=100, magnitude=0.5)])
    response = llama.generate(messages, max_new_tokens=50)
    print(f"Response: {response}")

    llama.clear_steering()
    print("\nDone.")
