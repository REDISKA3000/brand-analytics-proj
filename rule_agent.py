from __future__ import annotations

from typing import Any, Dict, List

from pydantic import BaseModel, Field


class DropCategoryDefinition(BaseModel):
    name: str
    description: str = ""


class RuleExample(BaseModel):
    text: str
    category: str


class GeneratedCategoryRules(BaseModel):
    category_name: str
    patterns: List[str] = Field(default_factory=list)


class RuleGenResult(BaseModel):
    rules: List[GeneratedCategoryRules] = Field(default_factory=list)


def _build_rules_prompt(
    profile: Dict[str, Any],
    drop_categories: List[Dict[str, Any]],
    examples: List[Dict[str, str]],
) -> str:
    brand_name = (profile.get("brand_name") or "BRAND").strip()
    desc = (profile.get("description") or "").strip()
    aliases = profile.get("aliases") or []
    aliases_str = ", ".join([a.strip() for a in aliases if str(a).strip()]) or "—"

    cat_lines = []
    for item in drop_categories:
        name = (item.get("name") or "").strip()
        if not name:
            continue
        description = (item.get("description") or "").strip()
        if description:
            cat_lines.append(f'- "{name}" — {description}')
        else:
            cat_lines.append(f'- "{name}"')
    cats_block = "\n".join(cat_lines) if cat_lines else "- (категории не заданы)"

    ex_lines = []
    for item in examples:
        text = (item.get("text") or "").strip()
        category = (item.get("category") or "").strip()
        if not text or not category:
            continue
        ex_lines.append(f'- "{text}" -> "{category}"')
    ex_block = "\n".join(ex_lines) if ex_lines else "- примеры не переданы"

    return f"""
Ты — помощник, который подбирает регулярные выражения (Python re) для фильтра релевантности бренда.
Нужно сгенерировать консервативные паттерны (минимум ложных drop) в пользовательских категориях:
{cats_block}

Бренд: {brand_name}
Описание: {desc if desc else "—"}
Алиасы: {aliases_str}

Правила генерации:
- Используй (?i) для регистронезависимости.
- Избегай слишком общих слов, которые часто встречаются в релевантных отзывах.
- Делай паттерны короткими и читаемыми.
- Не придумывай новые категории.
- Для каждой категории верни список regex именно для неё.
- Если по категории нет хороших консервативных regex, верни пустой список.
- Возвращай только JSON по схеме RuleGenResult.

Примеры:
{ex_block}
""".strip()


def generate_rules(
    profile: Dict[str, Any],
    drop_categories: List[Dict[str, Any]],
    examples: List[Dict[str, str]],
    *,
    client,
    model: str,
    temperature: float = 0.2,
) -> RuleGenResult:
    prompt = _build_rules_prompt(profile, drop_categories, examples)
    resp = client.responses.parse(
        model=model,
        input=[{"role": "user", "content": prompt}],
        text_format=RuleGenResult,
        temperature=temperature,
    )

    parsed: RuleGenResult = resp.output_parsed
    allowed = [(item.get("name") or "").strip() for item in drop_categories]
    allowed = [name for name in allowed if name]
    out_map = {name: [] for name in allowed}

    for rule_item in parsed.rules:
        name = (rule_item.category_name or "").strip()
        if name not in out_map:
            continue
        out_map[name] = [p.strip() for p in (rule_item.patterns or []) if str(p).strip()]

    ordered_rules = [
        GeneratedCategoryRules(category_name=name, patterns=out_map.get(name, []))
        for name in allowed
    ]
    return RuleGenResult(rules=ordered_rules)
