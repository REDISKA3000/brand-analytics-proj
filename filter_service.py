# filter_service.py
import time
import re
from typing import List, Tuple, Optional, Dict, Any, Callable
from concurrent.futures import ThreadPoolExecutor, as_completed

from llm_model import OpenAIRelevanceBatchModel


def chunk_list(lst, n: int):
    return [lst[i: i + n] for i in range(0, len(lst), n)]


def prepare_comment(text: str, truncate_chars: int = 800) -> str:
    s = "" if text is None else str(text)
    s = s.strip()
    if truncate_chars and len(s) > truncate_chars:
        s = s[:truncate_chars]
    return s


def _match_any(patterns: List[str], text: str) -> Optional[str]:
    for p in patterns or []:
        try:
            if re.search(p, text, flags=re.IGNORECASE):
                return p
        except re.error:
            # если пользователь ввёл кривой regex — просто игнорим конкретную строку
            continue
    return None


class RuleEngine:
    """
    Правила "точно drop" до LLM.
    Предпочтительный формат profile:
      - drop_categories: [{name, description, patterns}, ...]

    Для совместимости поддерживается и старый формат brands.yaml:
      - sure_drop_patterns
      - brand_sure_drop
      - pr_reply_markers
      - homonym_noise
      - search_noise_patterns
    """

    @staticmethod
    def _profile_categories(profile: Dict[str, Any]) -> List[Dict[str, Any]]:
        categories = []
        for item in profile.get("drop_categories", []) or []:
            if not isinstance(item, dict):
                continue
            name = (item.get("name") or item.get("category_name") or "").strip()
            description = (item.get("description") or "").strip()
            patterns = [str(p).strip() for p in (item.get("patterns") or []) if str(p).strip()]
            if not name:
                continue
            categories.append(
                {
                    "name": name,
                    "description": description,
                    "patterns": patterns,
                }
            )

        if any(item.get("patterns") for item in categories):
            return categories

        legacy_map = [
            ("sure_drop_patterns", "Вакансии и найм"),
            ("brand_sure_drop", "Локация и ориентир"),
            ("pr_reply_markers", "Официальный ответ"),
            ("homonym_noise", "Омонимы и другие бизнесы"),
            ("search_noise_patterns", "Поисковый и SEO шум"),
        ]
        for key, name in legacy_map:
            patterns = [str(p).strip() for p in (profile.get(key) or []) if str(p).strip()]
            if not patterns:
                continue
            categories.append({"name": name, "description": "", "patterns": patterns})
        return categories

    def sure_drop(self, text: str, profile: Dict[str, Any]) -> Optional[Dict[str, str]]:
        if not text:
            return None

        for category in self._profile_categories(profile):
            hit = _match_any(category.get("patterns", []), text)
            if hit:
                return {
                    "rule_code": category["name"],
                    "category_name": category["name"],
                    "pattern": hit,
                }

        return None


class RelevanceFilterService:
    """
    Основной сервис: подготовка текста, RULE, preprocess (внешний), LLM батчи, параллельность.
    preprocess_fn — функция, которую ты уже сделал через processing.py (preprocess_for_llm).
    """

    def __init__(self, llm: OpenAIRelevanceBatchModel, rule_engine: Optional[RuleEngine] = None):
        self.llm = llm
        self.rules = rule_engine or RuleEngine()

    def classify_one(
        self,
        raw_text: str,
        *,
        profile: Dict[str, Any],
        system_prompt: str,
        preprocess_fn,                 # (text_rule: str) -> text_llm: str
        truncate_chars: int = 800,
        model: Optional[str] = None,
        temperature: float = 0.0,
    ) -> Dict[str, Any]:
        """
        Возвращает dict:
          - action: keep/drop
          - source: rule/llm
          - latency_s (если llm)
          - rule (если rule)
        """
        text_rule = prepare_comment(raw_text, truncate_chars=truncate_chars)

        hit = self.rules.sure_drop(text_rule, profile)
        if hit is not None:
            return {"action": "drop", "source": "rule", "rule": hit}

        text_llm = preprocess_fn(text_rule)

        rows, dt = self.llm.classify_batch(
            batch=[(0, text_llm)],
            system_prompt=system_prompt,
            model=model,
            temperature=temperature,
        )
        return {"action": rows[0]["action"], "source": "llm", "latency_s": dt}

    def classify_many_parallel(
        self,
        texts: List[str],
        *,
        profile: Dict[str, Any],
        system_prompt: str,
        preprocess_fn,                 # (text_rule: str) -> text_llm: str
        batch_size: int = 6,
        max_workers: int = 3,
        truncate_chars: int = 800,
        model: Optional[str] = None,
        temperature: float = 0.0,
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
    ) -> tuple[List[str], Dict[str, Any]]:
        """
        Возвращает:
          actions: list[str] keep/drop по порядку
          stats: метрики
        """
        t0_total = time.perf_counter()

        text_rule_list = [prepare_comment(
            t, truncate_chars=truncate_chars) for t in texts]

        actions: List[Optional[str]] = [None] * len(text_rule_list)
        to_llm_pairs: List[Tuple[int, str]] = []

        rule_drop_count = 0
        for i, t_rule in enumerate(text_rule_list):
            hit = self.rules.sure_drop(t_rule, profile)
            if hit is not None:
                actions[i] = "drop"
                rule_drop_count += 1
            else:
                to_llm_pairs.append((i, preprocess_fn(t_rule)))

        if progress_callback is not None:
            progress_callback(rule_drop_count, len(text_rule_list), "relevance")

        if not to_llm_pairs:
            total_s = time.perf_counter() - t0_total
            if progress_callback is not None:
                progress_callback(len(text_rule_list), len(text_rule_list), "relevance")
            return actions, {
                "total_s": total_s,
                "n": len(text_rule_list),
                "rule_drops": rule_drop_count,
                "llm_calls": 0,
                "comments_per_s": (len(text_rule_list) / total_s) if total_s > 0 else None,
                "mean_batch_latency_s": None,
            }

        batches = chunk_list(to_llm_pairs, batch_size)

        llm_rows = []
        latencies = []
        completed_count = rule_drop_count

        def one_job(batch):
            return self.llm.classify_batch(
                batch=batch,
                system_prompt=system_prompt,
                model=model,
                temperature=temperature,
            )

        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futs = [ex.submit(one_job, b) for b in batches]
            for f in as_completed(futs):
                rows, dt = f.result()
                llm_rows.extend(rows)
                latencies.append(dt)
                completed_count += len(rows)
                if progress_callback is not None:
                    progress_callback(completed_count, len(text_rule_list), "relevance")

        for r in llm_rows:
            idx = int(r["global_idx"])
            actions[idx] = r["action"]

        # safety
        for i in range(len(actions)):
            if actions[i] is None:
                actions[i] = "keep"

        total_s = time.perf_counter() - t0_total
        if progress_callback is not None:
            progress_callback(len(text_rule_list), len(text_rule_list), "relevance")
        return actions, {
            "total_s": total_s,
            "n": len(text_rule_list),
            "rule_drops": rule_drop_count,
            "llm_calls": len(batches),
            "comments_per_s": (len(text_rule_list) / total_s) if total_s > 0 else None,
            "mean_batch_latency_s": (sum(latencies) / len(latencies)) if latencies else None,
        }
