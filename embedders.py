# embedders.py
from __future__ import annotations
from typing import List, Optional
import numpy as np


class OpenAIEmbedder:
    def __init__(self, client, model: str = "text-embedding-3-small", dimensions: Optional[int] = None):
        self.client = client
        self.model = model
        self.dimensions = dimensions

    def embed_texts(self, texts: List[str], batch_size: int = 128) -> np.ndarray:
        clean = ["" if t is None else str(t) for t in texts]
        out: List[List[float]] = []

        for i in range(0, len(clean), batch_size):
            batch = clean[i: i + batch_size]
            kwargs = {}
            if self.dimensions is not None:
                kwargs["dimensions"] = int(self.dimensions)

            resp = self.client.embeddings.create(
                model=self.model,
                input=batch,
                encoding_format="float",
                **kwargs,
            )
            data = sorted(resp.data, key=lambda x: x.index)
            out.extend([d.embedding for d in data])

        return np.asarray(out, dtype=np.float32)


class SentenceTransformerEmbedder:
    """
    Для локального обучения/инференса (GPU/CPU) через sentence-transformers.
    """

    def __init__(self, model_name: str, device: Optional[str] = None):
        from sentence_transformers import SentenceTransformer
        import torch

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model_name = model_name
        self.device = device
        self.model = SentenceTransformer(model_name).to(device)

    def embed_texts(self, texts: List[str], batch_size: int = 64) -> np.ndarray:
        clean = ["" if t is None else str(t) for t in texts]
        emb = self.model.encode(
            clean,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=False,
        )
        return emb.astype(np.float32)
