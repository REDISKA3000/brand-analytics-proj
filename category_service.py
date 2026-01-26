# category_service.py
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from category_model import CategoryIndex, CategoryTagger


def chunk_list(lst, n: int):
    return [lst[i:i+n] for i in range(0, len(lst), n)]


class CategoryTaggingService:
    def __init__(self, tagger: CategoryTagger):
        self.tagger = tagger

    def run(
        self,
        *,
        df_in: pd.DataFrame,
        text_col: str,
        user_prompt: str,
        preprocess_fn,
        ref_index: Optional[CategoryIndex] = None,
        is_drop_col: Optional[str] = "is_drop",
        top_k: int = 5,
        llm_batch_size: int = 10,
        max_workers: int = 3,
        truncate_chars: int = 800,
        embed_batch_size: int = 128,
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        if text_col not in df_in.columns:
            raise ValueError(f"Не найден столбец '{text_col}'")

        if not user_prompt or not str(user_prompt).strip():
            raise ValueError("Нужен prompt с определениями категорий")

        # подготовка
        raw_texts = df_in[text_col].astype(str).fillna("").tolist()
        clean_texts: List[str] = []
        for t in raw_texts:
            s = (t or "").strip()
            if truncate_chars and len(s) > int(truncate_chars):
                s = s[:int(truncate_chars)]
            # предобработка (НО в выходной файл не пишем)
            s = preprocess_fn(s)
            clean_texts.append(s)

        # skip mask (is_drop)
        skip = [False] * len(clean_texts)
        if is_drop_col and is_drop_col in df_in.columns:
            is_drop_vals = df_in[is_drop_col].astype(str).fillna("").tolist()
            for i, v in enumerate(is_drop_vals):
                if v.strip().lower() == "yes":
                    skip[i] = True

        # items to classify
        items_all: List[Tuple[int, str]] = []
        map_pos: List[int] = []
        for i, t in enumerate(clean_texts):
            if skip[i] or not t.strip():
                continue
            items_all.append((i, t))
            map_pos.append(i)

        # neighbors for RAG
        neighbors_all = None
        allowed_categories = None
        use_rag = ref_index is not None

        if use_rag:
            allowed_categories = ref_index.categories
            texts_for_neighbors = [t for _, t in items_all]
            neighbors_all = self.tagger.retrieve_neighbors(
                texts=texts_for_neighbors,
                index=ref_index,
                top_k=int(top_k),
                query_embed_batch_size=int(embed_batch_size),
            )

        # parallel LLM over chunks
        chunks = chunk_list(list(range(len(items_all))), int(llm_batch_size))
        preds = [None] * len(clean_texts)
        sources = ["skip_drop" if skip[i] else "skip_empty" if not clean_texts[i].strip(
        ) else "" for i in range(len(clean_texts))]

        t0 = time.perf_counter()

        def one_job(chunk_idxs: List[int]):
            batch_items = [items_all[j] for j in chunk_idxs]
            batch_neighbors = None
            if neighbors_all is not None:
                batch_neighbors = [neighbors_all[j] for j in chunk_idxs]

            out_map = self.tagger.classify_batch_llm(
                items=batch_items,
                user_prompt=user_prompt,
                allowed_categories=allowed_categories,
                neighbors_list=batch_neighbors,
            )
            return out_map

        with ThreadPoolExecutor(max_workers=int(max_workers)) as ex:
            futs = [ex.submit(one_job, ch) for ch in chunks]
            for f in as_completed(futs):
                out_map = f.result()
                for gi, cat in out_map.items():
                    preds[gi] = cat
                    sources[gi] = "rag_llm" if use_rag else "llm"

        total_s = time.perf_counter() - t0

        df_out = df_in.copy()
        df_out["Категория_pred"] = preds
        df_out["Категория_source"] = sources

        meta = {
            "total_s": total_s,
            "rows": len(df_in),
            "classified": sum(1 for x in sources if x in ("llm", "rag_llm")),
            "skipped_drop": sum(1 for x in sources if x == "skip_drop"),
            "use_rag": bool(use_rag),
            "top_k": int(top_k),
        }
        return df_out, meta
