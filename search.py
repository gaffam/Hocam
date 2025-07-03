import os
from typing import List, Tuple
import numpy as np
import faiss


class VectorSearch:
    def __init__(
        self,
        dim: int,
        grade: int,
        subject: str,
        base_dir: str = "index",
    ):
        os.makedirs(base_dir, exist_ok=True)
        subject_safe = subject.lower().replace(" ", "_")
        self.index_path = os.path.join(base_dir, f"{grade}_{subject_safe}.index")
        if os.path.exists(self.index_path):
            self.index = faiss.read_index(self.index_path)
        else:
            self.index = faiss.IndexFlatL2(dim)

    def add_embeddings(self, embeddings: np.ndarray):
        self.index.add(embeddings)
        faiss.write_index(self.index, self.index_path)

    def search(self, query_vec: np.ndarray, top_k: int = 3) -> Tuple[np.ndarray, np.ndarray]:
        distances, indices = self.index.search(query_vec.reshape(1, -1), top_k)
        return distances[0], indices[0]

    def search_with_rerank(
        self,
        query_vec: np.ndarray,
        embeddings: np.ndarray,
        top_k: int = 3,
        search_k: int = 10,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Search top ``search_k`` then re-rank by cosine similarity."""
        dist, idx = self.index.search(query_vec.reshape(1, -1), search_k)
        candidates = embeddings[idx[0]]
        sims = np.dot(candidates, query_vec) / (
            np.linalg.norm(candidates, axis=1) * np.linalg.norm(query_vec) + 1e-8
        )
        order = np.argsort(-sims)[:top_k]
        return sims[order], idx[0][order]
