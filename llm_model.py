# llm_model.py
import time
import random
import threading
from collections import deque
from typing import List, Literal, Tuple, Dict, Optional

from pydantic import BaseModel
from openai import OpenAI
import openai  # exceptions


class FilterItem(BaseModel):
    global_idx: int
    action: Literal["keep", "drop"]


class BatchResult(BaseModel):
    results: List[FilterItem]


class _RollingTokenLimiter:
    """
    Грубый локальный limiter по tokens-per-minute.
    Мы резервируем фиксированный бюджет на каждый запрос заранее, чтобы
    не ловить 429 на параллельных batch-вызовах.
    """

    def __init__(
        self,
        *,
        tokens_per_minute: int,
        estimated_tokens_per_request: int,
        window_seconds: float = 60.0,
    ):
        self.tokens_per_minute = int(tokens_per_minute)
        self.estimated_tokens_per_request = int(estimated_tokens_per_request)
        self.window_seconds = float(window_seconds)
        self._lock = threading.Lock()
        self._events = deque()
        self._used_tokens = 0

    def _drop_expired(self, now: float) -> None:
        while self._events and (now - self._events[0][0]) >= self.window_seconds:
            _, tokens = self._events.popleft()
            self._used_tokens -= tokens

    def acquire(self) -> None:
        reserve_tokens = max(1, self.estimated_tokens_per_request)

        while True:
            now = time.monotonic()
            with self._lock:
                self._drop_expired(now)
                if self._used_tokens + reserve_tokens <= self.tokens_per_minute:
                    self._events.append((now, reserve_tokens))
                    self._used_tokens += reserve_tokens
                    return

                oldest_ts = self._events[0][0]
                wait_s = max(0.05, self.window_seconds - (now - oldest_ts) + 0.05)

            time.sleep(wait_s)


class OpenAIRelevanceBatchModel:
    """
    Обёртка над OpenAI: structured output + ретраи + latency.
    """

    # Локальный хардкод под текущий runtime-лимит gpt-4.1.
    MODEL_TPM_LIMITS = {
        "gpt-4.1": 30000,
    }
    MODEL_ESTIMATED_TOKENS_PER_REQUEST = {
        # Консервативная оценка на один relevance batch.
        "gpt-4.1": 3000,
    }
    _LIMITERS: Dict[str, _RollingTokenLimiter] = {}
    _LIMITERS_LOCK = threading.Lock()

    def __init__(self, client: OpenAI, default_model: str = "gpt-5.4-mini"):
        self.client = client
        self.default_model = default_model

    @staticmethod
    def _reasoning_options(model_name: str) -> Optional[Dict[str, str]]:
        if str(model_name or "").startswith("gpt-5.4-mini"):
            return {"effort": "none"}
        return None

    @classmethod
    def _get_limiter(cls, model_name: str) -> Optional[_RollingTokenLimiter]:
        tpm = cls.MODEL_TPM_LIMITS.get(model_name)
        estimated = cls.MODEL_ESTIMATED_TOKENS_PER_REQUEST.get(model_name)
        if not tpm or not estimated:
            return None

        with cls._LIMITERS_LOCK:
            limiter = cls._LIMITERS.get(model_name)
            if limiter is None:
                limiter = _RollingTokenLimiter(
                    tokens_per_minute=tpm,
                    estimated_tokens_per_request=estimated,
                )
                cls._LIMITERS[model_name] = limiter
            return limiter

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
        limiter = self._get_limiter(use_model)
        reasoning = self._reasoning_options(use_model)

        for attempt in range(max_retries):
            try:
                if limiter is not None:
                    limiter.acquire()

                t0 = time.perf_counter()
                resp = self.client.responses.parse(
                    model=use_model,
                    input=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt},
                    ],
                    text_format=BatchResult,
                    temperature=temperature,
                    reasoning=reasoning,
                )
                dt = time.perf_counter() - t0

                parsed: BatchResult = resp.output_parsed
                rows = [{"global_idx": r.global_idx, "action": r.action}
                        for r in parsed.results]
                return rows, dt

            except openai.RateLimitError:
                # Если даже после локального throttling словили 429,
                # значит оценка токенов занижена или лимит уже съели другие запросы.
                if limiter is not None:
                    time.sleep(60.0)
                else:
                    sleep = min(8.0, 0.5 * (2 ** attempt)) + random.random() * 0.2
                    time.sleep(sleep)

            except (
                openai.APITimeoutError,
                openai.APIConnectionError,
                openai.InternalServerError,
            ):
                sleep = min(8.0, 0.5 * (2 ** attempt)) + random.random() * 0.2
                time.sleep(sleep)

        raise RuntimeError("Max retries exceeded for batch")
