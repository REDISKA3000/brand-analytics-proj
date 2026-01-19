# sentiment_service.py
from __future__ import annotations
from typing import Callable, Dict, List, Optional, Sequence, Tuple
import numpy as np


class SentimentService:
    def __init__(
        self,
        model,                      # SentimentModel
        # твоя предобработка (processing.py)
        preprocess_fn: Callable[[str], str],
        embed_batch_size: int = 128,
    ):
        self.model = model
        self.preprocess_fn = preprocess_fn
        self.embed_batch_size = embed_batch_size

    def predict_one(self, text: str) -> Dict:
        clean = self.preprocess_fn(text)
        if not clean:
            clean = "" if text is None else str(text)

        df = self.model.predict_texts(
            [clean], embed_batch_size=self.embed_batch_size, return_df=True)
        row = df.iloc[0].to_dict()
        return {
            "label": row["pred_label"],
            "source": row["pred_source"],
            "sim_pred": float(row["sim_pred"]),
            "scores": {
                "negative": float(row["s_negative"]),
                "neutral": float(row["s_neutral"]),
                "positive": float(row["s_positive"]),
            },
        }

    def predict_many(self, texts: Sequence[str]) -> Tuple[List[str], List[str]]:
        clean = [self.preprocess_fn(t) for t in texts]
        df = self.model.predict_texts(
            list(clean), embed_batch_size=self.embed_batch_size, return_df=True)
        labels = df["pred_label"].astype(str).tolist()
        sources = df["pred_source"].astype(str).tolist()
        return labels, sources
