# llm_model.py
import time
import random
from typing import List, Literal, Tuple, Dict, Optional

from pydantic import BaseModel
from openai import OpenAI
import openai  # exceptions


class FilterItem(BaseModel):
    global_idx: int
    action: Literal["keep", "drop"]


class BatchResult(BaseModel):
    results: List[FilterItem]


class OpenAIRelevanceBatchModel:
    """
    Обёртка над OpenAI: structured output + ретраи + latency.
    """

    def __init__(self, client: OpenAI, default_model: str = "gpt-4.1-mini"):
        self.client = client
        self.default_model = default_model

    @staticmethod
    def build_prompt(batch: List[Tuple[int, str]]) -> str:
        lines = [
            "Для каждой строки верни action=keep|drop.",
            "Сохраняй global_idx как есть. Верни ровно столько results, сколько входных строк.",
        ]
        for gi, txt in batch:
            lines.append(f"{gi}: {txt}")
        lines.append(
            "\nВерни JSON строго по схеме: {results: [{global_idx, action}, ...]}")
        return "\n".join(lines)

    def classify_batch(
        self,
        batch: List[Tuple[int, str]],
        *,
        system_prompt: str,
        model: Optional[str] = None,
        temperature: float = 0.0,
        max_retries: int = 6,
    ) -> tuple[List[Dict], float]:
        """
        Возвращает:
          rows: [{"global_idx": int, "action": "keep|drop"}...]
          latency_s: float (на весь батч)
        """
        prompt = self.build_prompt(batch)
        use_model = model or self.default_model

        for attempt in range(max_retries):
            try:
                t0 = time.perf_counter()
                resp = self.client.responses.parse(
                    model=use_model,
                    input=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt},
                    ],
                    text_format=BatchResult,
                    temperature=temperature,
                )
                dt = time.perf_counter() - t0

                parsed: BatchResult = resp.output_parsed
                rows = [{"global_idx": r.global_idx, "action": r.action}
                        for r in parsed.results]
                return rows, dt

            except (
                openai.RateLimitError,
                openai.APITimeoutError,
                openai.APIConnectionError,
                openai.InternalServerError,
            ):
                sleep = min(8.0, 0.5 * (2 ** attempt)) + random.random() * 0.2
                time.sleep(sleep)

        raise RuntimeError("Max retries exceeded for batch")
