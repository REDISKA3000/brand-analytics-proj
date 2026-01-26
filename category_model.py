# category_model.py
import json
import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def _norm(s: str) -> str:
    s = (s or "").lower().replace("ё", "е").strip()
    s = s.strip('"\''"«»`*_•-— ")
    s = re.sub(r"[^0-9a-zа-я\s]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _extract_json(text: str) -> Optional[dict]:
    if not text:
        return None
    m = _JSON_RE.search(text)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None


def _best_match_to_allowed(raw: str, allowed: Sequence[str]) -> Optional[str]:
    if not raw or not allowed:
        return None

    raw = raw.strip()
    if raw in allowed:
        return raw

    raw_n = _norm(raw)
    if not raw_n:
        return None

    best_cat = None
    best_score = 0.0

    for c in allowed:
        cn = _norm(c)
        if not cn:
            continue

        # prefix / substring
        if cn.startswith(raw_n) and len(raw_n) >= 4:
            return c
        if raw_n.startswith(cn) and len(cn) >= 4:
            return c

        score = SequenceMatcher(None, raw_n, cn).ratio()
        if score > best_score:
            best_score = score
            best_cat = c

    return best_cat if best_score >= 0.60 else None


@dataclass
class CategoryIndex:
    ref_texts: List[str]
    ref_cats: List[str]
    ref_emb: np.ndarray  # (N, D) normalized float32
    categories: List[str]


class CategoryTagger:
    """
    2 режима:
    - LLM-only: если нет ref-датасета
    - RAG: если ref-датасет есть -> retrieval top_k + LLM по user prompt
    Референсы: обязаны иметь Текст + Категория.
    """

    def __init__(
        self,
        *,
        client,
        embedder,
        llm_model: str = "gpt-4.1-mini",
        max_output_tokens: int = 1200,
        temperature: float = 0.0,
    ):
        self.client = client
        self.embedder = embedder
        self.llm_model = llm_model
        self.max_output_tokens = int(max_output_tokens)
        self.temperature = float(temperature)

    @staticmethod
    def _l2_normalize(x: np.ndarray) -> np.ndarray:
        x = x.astype(np.float32, copy=False)
        n = np.linalg.norm(x, axis=1, keepdims=True) + 1e-12
        return x / n

    def build_index(
        self,
        *,
        ref_texts: List[str],
        ref_cats: List[str],
        embed_batch_size: int = 128,
    ) -> CategoryIndex:
        if len(ref_texts) != len(ref_cats):
            raise ValueError("ref_texts and ref_cats must have same length")

        emb = self.embedder.embed_texts(
            ref_texts, batch_size=int(embed_batch_size))
        emb = self._l2_normalize(emb)

        cats = sorted(
            list({c for c in ref_cats if c is not None and str(c).strip()}))
        return CategoryIndex(
            ref_texts=ref_texts,
            ref_cats=ref_cats,
            ref_emb=emb,
            categories=cats,
        )

    def retrieve_neighbors(
        self,
        *,
        texts: List[str],
        index: CategoryIndex,
        top_k: int = 5,
        query_embed_batch_size: int = 128,
    ) -> List[List[Dict[str, Any]]]:
        if not texts:
            return []

        q = self.embedder.embed_texts(
            texts, batch_size=int(query_embed_batch_size))
        q = self._l2_normalize(q)

        # batched dot to avoid huge memory spikes
        neighbors_all: List[List[Dict[str, Any]]] = []
        ref = index.ref_emb
        n_ref = ref.shape[0]
        k = max(1, int(top_k))
        k = min(k, n_ref)

        for i in range(q.shape[0]):
            sims = ref @ q[i]  # (n_ref,)
            # top-k
            idx = np.argpartition(-sims, kth=k-1)[:k]
            idx = idx[np.argsort(-sims[idx])]

            nbs: List[Dict[str, Any]] = []
            for j in idx.tolist():
                nbs.append(
                    {
                        "text": index.ref_texts[j],
                        "category": index.ref_cats[j],
                        "similarity": float(sims[j]),
                    }
                )
            neighbors_all.append(nbs)

        return neighbors_all

    def _build_prompt_batch(
        self,
        *,
        items: List[Tuple[int, str]],
        user_prompt: str,
        allowed_categories: Optional[List[str]] = None,
        neighbors_list: Optional[List[List[Dict[str, Any]]]] = None,
        max_text_chars: int = 700,
        max_neighbor_chars: int = 240,
    ) -> str:
        assert items
        if neighbors_list is not None:
            assert len(neighbors_list) == len(items)

        allowed_block = ""
        if allowed_categories:
            allowed_block = "Разрешённые категории (выбирай строго из списка):\n" + "\n".join(
                f"- {c}" for c in allowed_categories
            )

        tasks = []
        for idx, (gi, txt) in enumerate(items):
            t = (txt or "").replace("\n", " ").strip()
            if len(t) > max_text_chars:
                t = t[:max_text_chars] + "..."

            examples_block = ""
            if neighbors_list is not None:
                examples = []
                for j, nb in enumerate(neighbors_list[idx]):
                    ex = (nb.get("text") or "").replace("\n", " ").strip()
                    if len(ex) > max_neighbor_chars:
                        ex = ex[:max_neighbor_chars] + "..."
                    examples.append(
                        f"[пример {j+1}] Категория: {nb.get('category')} | Текст: {ex}"
                    )
                examples_block = "\n".join(
                    examples) if examples else "(нет примеров)"

            block = f"""[ITEM {gi}]
Текст:
\"\"\"{t}\"\"\"
"""
            if neighbors_list is not None:
                block += f"Похожие примеры из предразметки:\n{examples_block}\n"
            tasks.append(block)

        tasks_block = "\n".join(tasks)

        prompt = f"""
{user_prompt.strip()}

{allowed_block}

ИНСТРУКЦИЯ:
- Для каждого ITEM выбери ОДНУ категорию по основной теме.
- Верни СТРОГО JSON без пояснений.

Формат:
{{
  "results": [
    {{"global_idx": 0, "category": "..." }},
    ...
  ]
}}

ITEM'ы:
{tasks_block}
""".strip()

        return prompt

    def classify_batch_llm(
        self,
        *,
        items: List[Tuple[int, str]],
        user_prompt: str,
        allowed_categories: Optional[List[str]] = None,
        neighbors_list: Optional[List[List[Dict[str, Any]]]] = None,
        system_hint: str = "Ты — аналитик отзывов. Следуй инструкции пользователя и верни строго JSON.",
    ) -> Dict[int, Optional[str]]:
        prompt = self._build_prompt_batch(
            items=items,
            user_prompt=user_prompt,
            allowed_categories=allowed_categories,
            neighbors_list=neighbors_list,
        )

        resp = self.client.responses.create(
            model=self.llm_model,
            input=[
                {"role": "system", "content": system_hint},
                {"role": "user", "content": prompt},
            ],
            max_output_tokens=self.max_output_tokens,
            temperature=self.temperature,
        )

        raw = resp.output[0].content[0].text if resp and resp.output else ""
        data = _extract_json(raw) or {}
        results = data.get("results", [])

        out: Dict[int, Optional[str]] = {gi: None for gi, _ in items}

        for r in results:
            try:
                gi = int(r.get("global_idx"))
                cat = (r.get("category") or "").strip()
            except Exception:
                continue
            if gi in out:
                if allowed_categories:
                    mapped = _best_match_to_allowed(cat, allowed_categories)
                    out[gi] = mapped or cat or None
                else:
                    out[gi] = cat or None

        # fallback: если LLM не вернул часть, оставим None
        return out
