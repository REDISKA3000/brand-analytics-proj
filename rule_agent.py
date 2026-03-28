from __future__ import annotations

from typing import Any, Dict, List

from pydantic import BaseModel, Field


class RuleGenResult(BaseModel):
    sure_drop_patterns: List[str] = Field(default_factory=list)
    brand_sure_drop: List[str] = Field(default_factory=list)
    homonym_noise: List[str] = Field(default_factory=list)
    search_noise_patterns: List[str] = Field(default_factory=list)
    pr_reply_markers: List[str] = Field(default_factory=list)


def _build_rules_prompt(profile: Dict[str, Any], examples: List[str]) -> str:
    brand_name = (profile.get("brand_name") or "BRAND").strip()
    desc = (profile.get("description") or "").strip()
    aliases = profile.get("aliases") or []
    aliases_str = ", ".join([a.strip() for a in aliases if str(a).strip()]) or "—"

    ex_block = ""
    if examples:
        ex_lines = [f"- {e}" for e in examples]
        ex_block = "Примеры пользовательских сообщений:\n" + "\n".join(ex_lines)

    return f"""
Ты — помощник, который подбирает регулярные выражения (Python re) для фильтра релевантности бренда.
Нужно сгенерировать консервативные паттерны (минимум ложных drop) в пяти категориях:
1) sure_drop_patterns — вакансии/найм
2) brand_sure_drop — локация/ориентир (косвенное упоминание)
3) homonym_noise — омонимы/другие бизнесы
4) search_noise_patterns — поисковый/SEO шум
5) pr_reply_markers — официальные ответы

Бренд: {brand_name}
Описание: {desc if desc else "—"}
Алиасы: {aliases_str}

Правила генерации:
- Используй (?i) для регистронезависимости.
- Избегай слишком общих слов, которые часто встречаются в релевантных отзывах.
- Давай короткие и читаемые regex.
- Возвращай только JSON по схеме RuleGenResult.

{ex_block}
""".strip()


def generate_rules(
    profile: Dict[str, Any],
    examples: List[str],
    *,
    client,
    model: str,
    temperature: float = 0.2,
) -> RuleGenResult:
    prompt = _build_rules_prompt(profile, examples)
    resp = client.responses.parse(
        model=model,
        input=[{"role": "user", "content": prompt}],
        text_format=RuleGenResult,
        temperature=temperature,
    )
    return resp.output_parsed
