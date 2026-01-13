import os
import time
import random
import re
from typing import List, Literal, Tuple, Optional, Dict, Any

import streamlit as st
import yaml
from pydantic import BaseModel
from openai import OpenAI
import openai  # exceptions
try:
    from config_local import OPENAI_API_KEY as LOCAL_OPENAI_API_KEY
except Exception:
    LOCAL_OPENAI_API_KEY = None


# ---------------- UI CONFIG ----------------
st.set_page_config(
    page_title="Relevance Filter",
    page_icon="🧼",
    layout="centered",
)

st.markdown(
    """
<style>
.block-container { padding-top: 2rem; max-width: 980px; }
.small-note { opacity: 0.75; font-size: 0.92rem; }
.card {
  background: white;
  border: 1px solid rgba(0,0,0,0.06);
  border-radius: 16px;
  padding: 16px 16px 10px 16px;
  box-shadow: 0 6px 20px rgba(0,0,0,0.04);
}
.badge-keep, .badge-drop, .badge-rule {
  display:inline-block;
  padding: 6px 10px;
  border-radius: 999px;
  font-weight: 800;
  font-size: 0.9rem;
  border: 1px solid rgba(0,0,0,0.08);
}
.badge-keep { background: rgba(34,197,94,0.12); }
.badge-drop { background: rgba(239,68,68,0.12); }
.badge-rule { background: rgba(59,130,246,0.12); }
</style>
""",
    unsafe_allow_html=True,
)

DEFAULT_MODEL = "gpt-4.1-mini"

# Это — базовый шаблон, а карточка бренда будет вставляться ниже автоматически.
BASE_SYSTEM_TEMPLATE = """
Ты — фильтр релевантности для бренда "{brand_name}" и его официальных цифровых каналов (если есть: приложение/сайт/бонусы).

Контекст бренда:
{brand_description}

Синонимы/алиасы (как пользователи могут писать бренд):
{brand_aliases}

Задача: для каждого сообщения выбрать только одно:
- "keep" — если сообщение относится к бренду/магазину/сети/приложению/бонусам/покупкам/ассортименту/скидкам/сервису.
- "drop" — если это явно НЕ про бренд как магазин/приложение/сервис.

ПРАВИЛО ПО УМОЛЧАНИЮ: если есть сомнения — выбирай "keep".

Важно:
- Возвращай строго JSON по схеме.
- Никаких пояснений, причин, текста — только JSON.
""".strip()


# ---------------- Structured Output ----------------
class FilterItem(BaseModel):
    global_idx: int
    action: Literal["keep", "drop"]


class BatchResult(BaseModel):
    results: List[FilterItem]


# ---------------- Brand Profiles ----------------
def load_brands(path: str = "brands.yaml") -> Dict[str, Dict[str, Any]]:
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    # нормализуем структуру
    out = {}
    for k, v in data.items():
        if not isinstance(v, dict):
            continue
        out[k] = v
        out[k].setdefault("brand_name", k)
        out[k].setdefault("description", "")
        out[k].setdefault("aliases", [])
        out[k].setdefault("sure_drop_patterns", [])
        out[k].setdefault("pr_reply_markers", [])
    return out


def format_system_prompt(base_template: str, profile: Dict[str, Any]) -> str:
    brand_name = profile.get("brand_name", "BRAND")
    desc = (profile.get("description") or "").strip()
    aliases = profile.get("aliases") or []
    aliases_str = ", ".join([a.strip()
                            for a in aliases if str(a).strip()]) or "—"

    return base_template.format(
        brand_name=brand_name,
        brand_description=desc if desc else "—",
        brand_aliases=aliases_str,
    ).strip()


# ---------------- API ----------------
def get_api_key() -> Optional[str]:
    secret_key = None
    try:
        secret_key = st.secrets.get("OPENAI_API_KEY", None)
    except Exception:
        # если secrets.toml не существует — Streamlit кидает FileNotFoundError
        secret_key = None

    return secret_key or os.getenv("OPENAI_API_KEY") or LOCAL_OPENAI_API_KEY


@st.cache_resource
def get_client() -> OpenAI:
    api_key = get_api_key()
    return OpenAI(api_key=api_key or "")


# ---------------- LLM helpers ----------------
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


def prepare_comment(text: str, truncate_chars: int = 800) -> str:
    s = "" if text is None else str(text)
    s = s.strip()
    if truncate_chars and len(s) > truncate_chars:
        s = s[:truncate_chars]
    return s


def classify_batch(
    batch: List[Tuple[int, str]],
    client: OpenAI,
    model: str,
    system_prompt: str,
    temperature: float = 0.0,
    max_retries: int = 6,
) -> Tuple[List[dict], float]:
    prompt = build_prompt(batch)

    last_err = None
    for attempt in range(max_retries):
        try:
            t0 = time.perf_counter()
            resp = client.responses.parse(
                model=model,
                input=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
                text_format=BatchResult,
                temperature=temperature,
            )
            dt = time.perf_counter() - t0

            parsed: BatchResult = resp.output_parsed
            out = []
            for r in parsed.results:
                out.append({"global_idx": r.global_idx,
                           "action": r.action, "batch_latency_s": dt})
            return out, dt

        except (
            openai.RateLimitError,
            openai.APITimeoutError,
            openai.APIConnectionError,
            openai.InternalServerError,
            openai.BadRequestError,
        ) as e:
            last_err = e
            sleep = min(8.0, 0.5 * (2**attempt)) + random.random() * 0.2
            time.sleep(sleep)

    raise RuntimeError(f"Max retries exceeded. Last error: {last_err!r}")


# ---------------- Pre-LLM "sure drop" rules ----------------
_PHONE_RE = re.compile(r"(?i)(\+?\d[\d\-\s\(\)]{8,}\d)")
# мягкий маркер “официального ответа”
_DEFAULT_PR_MARKERS = [
    r"(?i)^здравствуйте",
    r"(?i)^добрый\s+день",
    r"(?i)спасибо\s+за\s+(обращение|отзыв)",
    r"(?i)нам\s+очень\s+приятно",
    r"(?i)с\s+уважением",
]


def _match_any(patterns: List[str], text: str) -> Optional[str]:
    for p in patterns:
        try:
            if re.search(p, text):
                return p
        except re.error:
            # если кто-то положил плохую регэкспу — не падаем
            continue
    return None


def rule_based_sure_drop(text: str, profile: Dict[str, Any]) -> Optional[Dict[str, str]]:
    """
    Возвращает {"action":"drop","reason_code":"..."} если это ТОЧНО drop.
    Иначе None (пусть решает LLM).
    """
    t = (text or "").strip()
    if not t:
        return None

    # 1) PR/официальный ответ (очень часто начинается с приветствия)
    pr_markers = profile.get("pr_reply_markers") or []
    pr_hit = _match_any(pr_markers + _DEFAULT_PR_MARKERS, t)
    if pr_hit:
        return {"action": "drop", "reason_code": "pr_reply"}

    # 2) Хардовый найм: если есть ключевики + телефон/контакты — почти железно
    # (чтобы не резать обсуждения типа "как устроиться" без контактов — держим “строгим”)
    hiring_keywords = [
        r"(?i)\bваканси\w+\b",
        r"(?i)\bтребу(ется|ются)\b",
        r"(?i)\bподработк\w+\b",
        r"(?i)\bработа\b",
        r"(?i)\bнабор\b",
        r"(?i)\bсобеседовани\w+\b",
    ]
    if _match_any(hiring_keywords, t) and _PHONE_RE.search(t):
        return {"action": "drop", "reason_code": "hiring"}

    # 3) Бренд-специфичные “точно drop” паттерны из карточки
    sure_drop_patterns = profile.get("sure_drop_patterns") or []
    hit = _match_any(sure_drop_patterns, t)
    if hit:
        return {"action": "drop", "reason_code": "brand_sure_drop"}

    return None


# ---------------- UI ----------------
st.title("Relevance Filter")
st.caption("Карточка бренда + строгие правила «точно drop» перед LLM.")

brands = load_brands("brands.yaml")

with st.sidebar:
    st.subheader("Настройки")
    model = st.text_input("Model", value=DEFAULT_MODEL)
    temperature = st.slider("Temperature", 0.0, 1.0, 0.0, 0.1)
    truncate_chars = st.number_input(
        "Truncate chars", min_value=100, max_value=5000, value=800, step=50)

    brand_names = ["(manual)"] + sorted(list(brands.keys()))
    chosen = st.selectbox("Компания", brand_names, index=0)

    st.markdown('<div class="small-note">Под «manual» можно вставить карточку бренда руками.</div>',
                unsafe_allow_html=True)

api_key_present = bool(get_api_key())
if not api_key_present:
    st.warning(
        "Не найден OPENAI_API_KEY. Добавь ключ в env или Streamlit secrets.")

# Профиль бренда (из файла или ручной)
if chosen != "(manual)" and chosen in brands:
    profile = dict(brands[chosen])
else:
    profile = {
        "brand_name": st.session_state.get("manual_brand_name", "BRAND"),
        "description": st.session_state.get("manual_description", ""),
        "aliases": st.session_state.get("manual_aliases", []),
        "sure_drop_patterns": st.session_state.get("manual_sure_drop_patterns", []),
        "pr_reply_markers": st.session_state.get("manual_pr_reply_markers", []),
    }

# --- редактирование карточки бренда в UI ---
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("Карточка бренда")

col1, col2 = st.columns([1, 1])
with col1:
    brand_name = st.text_input(
        "Brand name", value=profile.get("brand_name", "BRAND"))
with col2:
    aliases_raw = st.text_input(
        "Aliases (через запятую)",
        value=", ".join(profile.get("aliases") or []),
        placeholder="familia, фамилия, ...",
    )

description = st.text_area(
    "Описание бренда (контекст)",
    value=profile.get("description", ""),
    height=140,
    placeholder="Кто это, что продаёт/делает, какие каналы (приложение/сайт), что считаем релевантным…",
)

with st.expander("Правила «точно drop» (бренд-специфичные регэкспы)"):
    sure_drop_list = profile.get("sure_drop_patterns") or []
    sure_drop_text = st.text_area(
        "По одному паттерну в строке",
        value="\n".join(sure_drop_list),
        height=140,
        placeholder=r'(?i)\bsagrada\s+familia\b',
    )

with st.expander("Маркеры PR/официальных ответов (регэкспы)"):
    pr_list = profile.get("pr_reply_markers") or []
    pr_text = st.text_area(
        "По одному паттерну в строке",
        value="\n".join(pr_list),
        height=120,
        placeholder=r"(?i)^здравствуйте",
    )

# обновим профиль из UI
profile["brand_name"] = brand_name.strip() if brand_name.strip() else "BRAND"
profile["aliases"] = [a.strip() for a in aliases_raw.split(",") if a.strip()]
profile["description"] = description.strip()
profile["sure_drop_patterns"] = [line.strip()
                                 for line in sure_drop_text.splitlines() if line.strip()]
profile["pr_reply_markers"] = [line.strip()
                               for line in pr_text.splitlines() if line.strip()]

# если manual — запомним в session_state, чтобы не терялось
if chosen == "(manual)":
    st.session_state["manual_brand_name"] = profile["brand_name"]
    st.session_state["manual_aliases"] = profile["aliases"]
    st.session_state["manual_description"] = profile["description"]
    st.session_state["manual_sure_drop_patterns"] = profile["sure_drop_patterns"]
    st.session_state["manual_pr_reply_markers"] = profile["pr_reply_markers"]

st.markdown("</div>", unsafe_allow_html=True)

# --- system prompt template ---
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("System prompt template (шаблон)")

base_template = st.text_area(
    "Шаблон (можно править)",
    value=st.session_state.get("base_template", BASE_SYSTEM_TEMPLATE),
    height=220,
)
st.session_state["base_template"] = base_template

final_system = format_system_prompt(base_template, profile)

with st.expander("Preview: итоговый system prompt"):
    st.code(final_system, language="text")

st.markdown("</div>", unsafe_allow_html=True)

# --- input comment + run ---
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("Комментарий (один)")

comment = st.text_area(
    "Вставь текст",
    value=st.session_state.get("comment", ""),
    height=140,
    placeholder="Один комментарий сюда…",
)
st.session_state["comment"] = comment

run = st.button("🚀 Запустить", type="primary", use_container_width=True)
st.markdown("</div>", unsafe_allow_html=True)

# --- run logic ---
if run:
    if not api_key_present:
        st.error("Нет OPENAI_API_KEY — добавь ключ и перезапусти приложение.")
        st.stop()

    c = prepare_comment(comment, truncate_chars=truncate_chars)
    if not c:
        st.error("Комментарий пустой — вставь текст.")
        st.stop()

    # 1) Pre-LLM sure-drop rules
    rule_hit = rule_based_sure_drop(c, profile)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Результат")

    if rule_hit is not None:
        # строгое drop без LLM
        st.markdown(
            '<span class="badge-drop">DROP</span> <span class="badge-rule">RULE</span>', unsafe_allow_html=True)
        st.caption(f"Pre-LLM правило сработало: {rule_hit['reason_code']}")
        st.write("JSON:")
        st.json({"results": [{"global_idx": 0, "action": "drop"}]})
        st.markdown("</div>", unsafe_allow_html=True)
        st.stop()

    # 2) LLM classification
    client = get_client()
    with st.spinner("Классифицирую через LLM…"):
        try:
            rows, dt = classify_batch(
                batch=[(0, c)],
                client=client,
                model=model,
                system_prompt=final_system,
                temperature=temperature,
                max_retries=6,
            )
            action = rows[0]["action"]

            if action == "keep":
                st.markdown('<span class="badge-keep">KEEP</span>',
                            unsafe_allow_html=True)
            else:
                st.markdown('<span class="badge-drop">DROP</span>',
                            unsafe_allow_html=True)

            st.caption(f"Latency: {dt:.3f}s")
            st.write("JSON:")
            st.json({"results": [{"global_idx": 0, "action": action}]})

        except Exception as e:
            st.error("Ошибка при запросе/парсинге.")
            st.exception(e)

    st.markdown("</div>", unsafe_allow_html=True)
