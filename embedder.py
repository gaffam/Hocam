from typing import List
import numpy as np
from sentence_transformers import SentenceTransformer


class Embedder:
    def __init__(
        self,
        model_name: str = "emrecan/bert-base-turkish-cased-uncased-nli-stsb-tr",
        cache_dir: str = None,
    ):
        """Load SentenceTransformer model. ``model_name`` can be changed if
        you want to try other embedding models."""

        self.model_name = model_name
        self.model = SentenceTransformer(model_name, cache_folder=cache_dir)

    def encode_texts(self, texts: List[str]) -> np.ndarray:
        return np.array(self.model.encode(texts, show_progress_bar=True, convert_to_numpy=True))

    def encode_query(self, query: str) -> np.ndarray:
        return self.model.encode([query], convert_to_numpy=True)[0]
