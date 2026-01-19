# SentimentModel: multi-centroid (k=3) + per-class thresholds + ChatGPT fallback
# deps: numpy, pandas, scikit-learn, openai, pydantic

import os
import time
import random
import threading
from dataclasses import dataclass
from typing import Callable, Dict, List, Literal, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from pydantic import BaseModel
from concurrent.futures import ThreadPoolExecutor, as_completed

from openai import OpenAI
import openai  # exceptions


def _to_np_f32(x) -> np.ndarray:
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    return np.asarray(x, dtype=np.float32)


def _l2_normalize(X: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(X, axis=1, keepdims=True)
    return X / (n + eps)


class _LLMItem(BaseModel):
    global_idx: int
    label: Literal["negative", "neutral", "positive"]


class _LLMBatch(BaseModel):
    results: List[_LLMItem]


@dataclass
class SentimentModelConfig:
    k_per_class: int = 3
    normalize: bool = True

    # thresholds per predicted class
    thresholds: Dict[int, float] = None  # {0:...,1:...,2:...}

    # fallback to LLM
    enable_llm_fallback: bool = True
    llm_model: str = "gpt-4.1-mini"
    llm_batch_size: int = 20
    llm_workers: int = 3
    truncate_chars: int = 800
    max_retries: int = 6


class SentimentModel:
    """
    1) fit(): строит k прототипов (центроидов) на класс (KMeans по каждому классу)
    2) predict_from_embeddings(): предсказывает по близости к прототипам
    3) predict_texts()/predict_df(): если не прошло пороги -> LLM fallback (опционально)
    """

    ID2LABEL_3 = {0: "negative", 1: "neutral", 2: "positive"}
    LABEL2ID_3 = {"negative": 0, "neutral": 1, "positive": 2}

    SYSTEM_LLM = (
        "Ты классификатор тональности русскоязычных отзывов/комментариев.\n"
        "Для каждого текста верни только label: negative | neutral | positive.\n"
        "Никаких пояснений и лишнего текста — только JSON по схеме."
    )

    def __init__(
        self,
        # embed_texts(texts, batch_size=...)
        embed_fn: Callable[[List[str], int], "Any"],
        config: Optional[SentimentModelConfig] = None,
        openai_api_key: Optional[str] = None,
    ):
        self.embed_fn = embed_fn
        self.cfg = config or SentimentModelConfig()
        if self.cfg.thresholds is None:
            # разумный дефолт; ты можешь передать свои
            self.cfg.thresholds = {0: 0.7, 1: 0.6, 2: 0.6}

        self._C: Optional[np.ndarray] = None
        self._proto2class: Optional[np.ndarray] = None

        # LLM client handling (thread-local)
        self._api_key = openai_api_key or os.environ.get(
            "OPENAI_API_KEY", None)
        self._tls = threading.local()

    # ----------------------------
    # training (prototypes)
    # ----------------------------
    def fit(self, X_train_emb, y_train) -> "SentimentModel":
        from sklearn.cluster import KMeans
        X = _to_np_f32(X_train_emb)
        y = np.asarray(y_train, dtype=np.int64)

        if self.cfg.normalize:
            X = _l2_normalize(X)

        classes = np.unique(y)
        centroids = []
        proto2class = []

        for c in classes:
            Xc = X[y == c]
            if len(Xc) == 0:
                continue

            k = min(self.cfg.k_per_class, len(Xc))
            if k == 1:
                Cc = Xc.mean(axis=0, keepdims=True)
            else:
                km = KMeans(n_clusters=k, random_state=42, n_init="auto")
                km.fit(Xc)
                Cc = km.cluster_centers_.astype(np.float32)

            if self.cfg.normalize:
                Cc = _l2_normalize(Cc)

            centroids.append(Cc)
            proto2class.append(np.full((Cc.shape[0],), c, dtype=np.int64))

        self._C = np.vstack(centroids).astype(np.float32)
        self._proto2class = np.concatenate(proto2class)
        return self

    # ----------------------------
    # prototype inference
    # ----------------------------
    def _check_fitted(self):
        if self._C is None or self._proto2class is None:
            raise RuntimeError(
                "Model is not fitted. Call fit(X_train, y_train) first.")

    def predict_from_embeddings(
        self,
        X_query_emb,
        return_scores: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Returns:
          pred3: (N,) predicted class id 0/1/2 (argmax)
          sim_pred: (N,) similarity for predicted class
          scores: (N,3) class scores (max similarity among prototypes of that class)
        """
        self._check_fitted()

        Xq = _to_np_f32(X_query_emb)
        if self.cfg.normalize:
            Xq = _l2_normalize(Xq)

        sims = Xq @ self._C.T  # (N, M)

        n = Xq.shape[0]
        scores = np.full((n, 3), -1e9, dtype=np.float32)
        for c in (0, 1, 2):
            m = (self._proto2class == c)
            if m.any():
                scores[:, c] = sims[:, m].max(axis=1)

        pred3 = scores.argmax(axis=1)
        sim_pred = scores[np.arange(n), pred3]
        return pred3, sim_pred, scores

    def apply_thresholds(
        self,
        pred3: np.ndarray,
        sim_pred: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Per-class thresholds. If not passed -> another(=3).
        Returns:
          pred4: (N,) 0/1/2 or 3
          pass_mask: (N,) bool
        """
        th = self.cfg.thresholds
        th_vec = np.array([th[0], th[1], th[2]], dtype=np.float32)
        pass_mask = sim_pred >= th_vec[pred3]
        pred4 = np.where(pass_mask, pred3, 3)
        return pred4, pass_mask

    # ----------------------------
    # LLM fallback
    # ----------------------------
    def _get_llm_client(self) -> OpenAI:
        if not self._api_key:
            raise RuntimeError(
                "OPENAI_API_KEY is not set. Set env OPENAI_API_KEY or pass openai_api_key to SentimentModel."
            )
        if not hasattr(self._tls, "client"):
            self._tls.client = OpenAI(api_key=self._api_key)
        return self._tls.client

    def _build_llm_prompt(self, batch: List[Tuple[int, str]]) -> str:
        lines = [
            "Определи тональность каждого пункта.",
            "Верни JSON строго по схеме: {results:[{global_idx,label},...]}",
        ]
        for gi, txt in batch:
            lines.append(f"{gi}: {txt}")
        return "\n".join(lines)

    def _llm_classify_batch(self, batch: List[Tuple[int, str]]) -> Dict[int, str]:
        client = self._get_llm_client()
        prompt = self._build_llm_prompt(batch)

        for attempt in range(self.cfg.max_retries):
            try:
                resp = client.responses.parse(
                    model=self.cfg.llm_model,
                    input=[
                        {"role": "system", "content": self.SYSTEM_LLM},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0,
                    text_format=_LLMBatch,
                )
                parsed: _LLMBatch = resp.output_parsed
                return {item.global_idx: item.label for item in parsed.results}

            except (openai.RateLimitError, openai.APITimeoutError, openai.APIConnectionError, openai.InternalServerError):
                sleep = min(8.0, 0.5 * (2**attempt)) + random.random() * 0.2
                time.sleep(sleep)

        raise RuntimeError("Max retries exceeded (LLM batch)")

    def llm_fallback_parallel(self, idx_and_text: List[Tuple[int, str]]) -> Dict[int, str]:
        if not idx_and_text:
            return {}

        bs = max(1, int(self.cfg.llm_batch_size))
        batches = [idx_and_text[i: i + bs]
                   for i in range(0, len(idx_and_text), bs)]

        out: Dict[int, str] = {}
        with ThreadPoolExecutor(max_workers=max(1, int(self.cfg.llm_workers))) as ex:
            futs = [ex.submit(self._llm_classify_batch, b) for b in batches]
            for f in as_completed(futs):
                out.update(f.result())
        return out

    # ----------------------------
    # High-level APIs
    # ----------------------------
    def predict_texts(
        self,
        texts: Sequence[str],
        embed_batch_size: int = 128,
        return_df: bool = True,
    ):
        texts = ["" if t is None else str(t).strip() for t in texts]
        # truncate for safety
        if self.cfg.truncate_chars:
            texts = [t[: self.cfg.truncate_chars] for t in texts]

        X = self.embed_fn(list(texts), batch_size=embed_batch_size)
        pred3, sim_pred, scores = self.predict_from_embeddings(X)
        pred4, pass_mask = self.apply_thresholds(pred3, sim_pred)

        labels = np.array([self.ID2LABEL_3[i] for i in pred3], dtype=object)
        source = np.where(pass_mask, "prototype", "llm")

        # LLM fallback
        if self.cfg.enable_llm_fallback:
            need = np.where(~pass_mask)[0].tolist()
            if need:
                payload = [(i, texts[i]) for i in need]
                llm_map = self.llm_fallback_parallel(payload)
                for i in need:
                    labels[i] = llm_map[i]  # must exist

        if not return_df:
            return labels, source, sim_pred, scores

        df = pd.DataFrame(
            {
                "text": texts,
                "pred_label": labels,
                "pred_source": source,
                "sim_pred": sim_pred,
                "s_negative": scores[:, 0],
                "s_neutral": scores[:, 1],
                "s_positive": scores[:, 2],
            }
        )
        return df

    def predict_df(
        self,
        df: pd.DataFrame,
        text_col: str = "clean_text",
        embed_batch_size: int = 128,
        inplace: bool = False,
    ) -> pd.DataFrame:
        if text_col not in df.columns:
            raise KeyError(
                f"Column '{text_col}' not found in df. Available: {list(df.columns)[:30]}")

        texts = df[text_col].astype(str).tolist()
        out = self.predict_texts(
            texts, embed_batch_size=embed_batch_size, return_df=True)

        if inplace:
            df["pred_label"] = out["pred_label"].values
            df["pred_source"] = out["pred_source"].values
            df["sim_pred"] = out["sim_pred"].values
            df["s_negative"] = out["s_negative"].values
            df["s_neutral"] = out["s_neutral"].values
            df["s_positive"] = out["s_positive"].values
            return df

        df2 = df.copy()
        df2["pred_label"] = out["pred_label"].values
        df2["pred_source"] = out["pred_source"].values
        df2["sim_pred"] = out["sim_pred"].values
        df2["s_negative"] = out["s_negative"].values
        df2["s_neutral"] = out["s_neutral"].values
        df2["s_positive"] = out["s_positive"].values
        return df2

    @property
    def is_fitted(self) -> bool:
        return self._C is not None and self._proto2class is not None

    def save_artifacts(self, path: str) -> None:
        """
        Сохраняет прототипы и пороги в .npz
        """
        self._check_fitted()
        th = self.cfg.thresholds or {0: 0.7, 1: 0.6, 2: 0.6}
        th_arr = np.array([th[0], th[1], th[2]], dtype=np.float32)

        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        np.savez_compressed(
            path,
            C=self._C.astype(np.float32),
            proto2class=self._proto2class.astype(np.int64),
            thresholds=th_arr,
            k_per_class=int(self.cfg.k_per_class),
            normalize=bool(self.cfg.normalize),
        )

    def load_artifacts(self, path: str) -> "SentimentModel":
        """
        Загружает прототипы и пороги из .npz
        """
        z = np.load(path, allow_pickle=False)
        self._C = z["C"].astype(np.float32)
        self._proto2class = z["proto2class"].astype(np.int64)

        th_arr = z.get("thresholds", None)
        if th_arr is not None:
            th_arr = th_arr.astype(np.float32).tolist()
            self.cfg.thresholds = {0: float(th_arr[0]), 1: float(
                th_arr[1]), 2: float(th_arr[2])}

        # если хочешь — подтягивать и эти поля тоже:
        if "k_per_class" in z:
            self.cfg.k_per_class = int(z["k_per_class"])
        if "normalize" in z:
            self.cfg.normalize = bool(z["normalize"])

        return self
