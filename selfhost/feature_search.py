"""
Feature label search index using sentence-transformers embeddings.
Loads 65k feature labels, embeds them, enables cosine similarity search.
"""

import json
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional


class FeatureSearchIndex:
    """Search SAE feature labels by semantic similarity."""

    def __init__(self, labels_path: str, model_name: str = "all-MiniLM-L6-v2"):
        """Load labels and build embedding index.

        Args:
            labels_path: Path to JSON file mapping index -> label string.
                         Supports both {"0": "label", ...} and [{"index_in_sae": 0, "label": "..."}] formats.
            model_name: Sentence transformer model for embeddings.
        """
        self.labels = self._load_labels(labels_path)
        self.indices = sorted(self.labels.keys())
        self.label_list = [self.labels[i] for i in self.indices]

        # Build embeddings
        print(f"Embedding {len(self.label_list)} feature labels...")
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name)
        self.embeddings = self.model.encode(self.label_list, show_progress_bar=True, normalize_embeddings=True)
        print(f"Embedding index ready. Shape: {self.embeddings.shape}")

    def _load_labels(self, path: str) -> dict:
        """Load feature labels from JSON file."""
        with open(path) as f:
            data = json.load(f)

        if isinstance(data, dict):
            # {"0": "label", "1": "label", ...}
            return {int(k): v for k, v in data.items()}
        elif isinstance(data, list):
            # [{"index_in_sae": 0, "label": "..."}, ...]
            return {item["index_in_sae"]: item["label"] for item in data}
        else:
            raise ValueError(f"Unexpected labels format: {type(data)}")

    def get_label(self, index: int) -> str:
        """Get label for a feature index."""
        return self.labels.get(index, f"feature_{index}")

    def search(self, query: str, top_k: int = 10) -> List[Tuple[int, str, float]]:
        """Search for features similar to query.

        Returns: List of (index, label, similarity) tuples.
        """
        query_emb = self.model.encode([query], normalize_embeddings=True)
        similarities = (query_emb @ self.embeddings.T)[0]

        top_idx = np.argsort(similarities)[-top_k:][::-1]
        results = []
        for i in top_idx:
            feat_idx = self.indices[i]
            results.append((feat_idx, self.label_list[i], float(similarities[i])))

        return results
