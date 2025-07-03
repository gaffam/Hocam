import os
from typing import Tuple
import numpy as np

# Choose backend with environment variable or default to FAISS
SEARCH_BACKEND = os.environ.get("SEARCH_BACKEND", "faiss").lower()

if SEARCH_BACKEND == "faiss":
    import faiss
elif SEARCH_BACKEND == "annoy":
    from annoy import AnnoyIndex
elif SEARCH_BACKEND == "scann":
    import scann
else:
    raise ValueError("Ge\u00e7ersiz SEARCH_BACKEND ayar\u0131.")

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

        self.backend = SEARCH_BACKEND
        suffix = {"faiss": "index", "annoy": "ann", "scann": "npy"}.get(self.backend, "index")
        self.index_path = os.path.join(base_dir, f"{grade}_{subject_safe}.{suffix}")
        self.dim = dim

        if self.backend == "faiss":
            if os.path.exists(self.index_path):
                self.index = faiss.read_index(self.index_path)
            else:
                self.index = faiss.IndexFlatL2(dim)
        elif self.backend == "annoy":
            self.index = AnnoyIndex(dim, "angular")
            if os.path.exists(self.index_path):
                self.index.load(self.index_path)
                self.counter = self.index.get_n_items()
            else:
                self.counter = 0
        elif self.backend == "scann":
            self.searcher = None
            self.embeddings = None
            if os.path.exists(self.index_path):
                self.embeddings = np.load(self.index_path)
                self.searcher = scann.scann_ops_pybind.builder(self.embeddings, 10, "dot_product").build()
        else:
            raise ValueError("Unsupported SEARCH_BACKEND")

    def add_embeddings(self, embeddings: np.ndarray):
        if self.backend == "faiss":
            self.index.add(embeddings)
            faiss.write_index(self.index, self.index_path)
        elif self.backend == "annoy":
            for vec in embeddings:
                self.index.add_item(self.counter, vec.tolist())
                self.counter += 1
            self.index.build(10)
            self.index.save(self.index_path)
        elif self.backend == "scann":
            self.embeddings = (
                embeddings
                if self.embeddings is None
                else np.vstack([self.embeddings, embeddings])
            )
            self.searcher = scann.scann_ops_pybind.builder(
                self.embeddings, 10, "dot_product"
            ).build()
            np.save(self.index_path, self.embeddings)
        else:
            raise ValueError("Unsupported SEARCH_BACKEND")

    def search(
        self, query_vec: np.ndarray, top_k: int = 3
    ) -> Tuple[np.ndarray, np.ndarray]:
        if self.backend == "faiss":
            dist, idx = self.index.search(query_vec.reshape(1, -1), top_k)
            return dist[0], idx[0]
        elif self.backend == "annoy":
            idx = self.index.get_nns_by_vector(
                query_vec.tolist(), top_k, include_distances=True
            )
            return np.array(idx[1]), np.array(idx[0])
        elif self.backend == "scann":
            if self.searcher is None:
                raise ValueError("ScaNN searcher not initialized")
            idx, dist = self.searcher.search(query_vec)
            return np.array(dist)[:top_k], np.array(idx)[:top_k]
        else:
            raise ValueError("Unsupported SEARCH_BACKEND")
=======
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

        """Search top ``search_k`` then re-rank by cosine similarity (FAISS only)."""
        if self.backend == "faiss":
            dist, idx = self.index.search(query_vec.reshape(1, -1), search_k)
            candidates = embeddings[idx[0]]
            sims = np.dot(candidates, query_vec) / (
                np.linalg.norm(candidates, axis=1) * np.linalg.norm(query_vec) + 1e-8
            )
            order = np.argsort(-sims)[:top_k]
            return sims[order], idx[0][order]
        else:
            return self.search(query_vec, top_k)

        """Search top ``search_k`` then re-rank by cosine similarity."""
        dist, idx = self.index.search(query_vec.reshape(1, -1), search_k)
        candidates = embeddings[idx[0]]
        sims = np.dot(candidates, query_vec) / (
            np.linalg.norm(candidates, axis=1) * np.linalg.norm(query_vec) + 1e-8
        )
        order = np.argsort(-sims)[:top_k]
        return sims[order], idx[0][order]

