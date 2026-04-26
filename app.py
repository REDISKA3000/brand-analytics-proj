# app.py
from __future__ import annotations

import hashlib
import io
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st
import yaml
from openai import OpenAI

import processing as proc  # processing.py рядом с app.py
from rule_agent import generate_rules

from embedders import OpenAIEmbedder
from sentiment_model import SentimentModel, SentimentModelConfig
from sentiment_service import SentimentService

# ВАЖНО: эти два файла должны лежать рядом с app.py:
# - llm_model.py  (класс OpenAIRelevanceBatchModel)
# - filter_service.py (класс RelevanceFilterService)
try:
    from llm_model import OpenAIRelevanceBatchModel
    from filter_service import RelevanceFilterService
except Exception as e:
    OpenAIRelevanceBatchModel = None
    RelevanceFilterService = None
    _IMPORT_ERR = e
else:
    _IMPORT_ERR = None

# Новые файлы для категорий:
# - category_model.py (CategoryTagger, CategoryIndex)
# - category_service.py (CategoryTaggingService)
try:
    from category_model import CategoryTagger, CategoryIndex
    from category_service import CategoryTaggingService
except Exception as e:
    CategoryTagger = None
    CategoryIndex = None
    CategoryTaggingService = None
    _CAT_IMPORT_ERR = e
else:
    _CAT_IMPORT_ERR = None

try:
    from config_local import OPENAI_API_KEY as LOCAL_OPENAI_API_KEY
except Exception:
    LOCAL_OPENAI_API_KEY = None

# ---------------- UI CONFIG ----------------
st.set_page_config(
    page_title="Brand Analytics",
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
  margin-bottom: 14px;
}

.badge-row {
  display: flex;
  flex-wrap: wrap;
  gap: 10px;
  align-items: center;
  margin-top: 8px;
  margin-bottom: 10px;
}

.badge {
  display: inline-flex;
  align-items: center;
  padding: 6px 12px;
  border-radius: 999px;
  font-weight: 800;
  font-size: 0.90rem;
  border: 1px solid rgba(0,0,0,0.08);
  line-height: 1;
}

.badge--green { background: rgba(34,197,94,0.12); color: #1b4332; }
.badge--red   { background: rgba(239,68,68,0.12); color: #7f1d1d; }
.badge--gray  { background: rgba(107,114,128,0.12); color: #374151; }
.badge--blue  { background: rgba(59,130,246,0.12); color: #1e3a8a; }

hr { border: none; border-top: 1px solid rgba(0,0,0,0.08); margin: 10px 0; }
</style>
""",
    unsafe_allow_html=True,
)

DEFAULT_LLM_MODEL = "gpt-4.1"
DEFAULT_EMBED_MODEL = "text-embedding-3-small"
DEFAULT_SENTIMENT_ARTIFACTS = "sentiment_assets/sentiment_openai.npz"

BASE_SYSTEM_TEMPLATE = """
Ты — фильтр релевантности для бренда "{brand_name}" и его официальных цифровых каналов (если есть: приложение/сайт/бонусы).

Контекст бренда:
{brand_description}

Синонимы/алиасы (как пользователи могут писать бренд):
{brand_aliases}

Категории, которые обычно считаются KEEP:
{brand_keep_categories}

Категории, которые обычно считаются DROP:
{brand_drop_categories}

Задача: для каждого сообщения выбрать только одно:
- "keep" — если сообщение относится к бренду и попадает в keep-категории или в близкий к ним пользовательский сценарий.
- "drop" — если сообщение явно не относится к бренду или попадает в drop-категории.

ПРАВИЛО ПО УМОЛЧАНИЮ: если есть сомнения — выбирай "keep".

Важно:
- Возвращай строго JSON по схеме.
- Никаких пояснений, причин, текста — только JSON.
""".strip()


# ---------------- Helpers: brands ----------------
def load_brands(path: str = "brands.yaml") -> Dict[str, Dict[str, Any]]:
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    out: Dict[str, Dict[str, Any]] = {}
    for k, v in data.items():
        if not isinstance(v, dict):
            continue
        p = dict(v)
        p.setdefault("brand_name", k)
        p.setdefault("description", "")
        p.setdefault("aliases", [])
        p.setdefault("keep_categories", [])
        p.setdefault("drop_categories", [])
        p.setdefault("sure_drop_patterns", [])
        p.setdefault("pr_reply_markers", [])
        # совместимость с RuleEngine из filter_service.py
        p.setdefault("brand_sure_drop", [])
        p.setdefault("homonym_noise", [])
        p.setdefault("search_noise_patterns", [])
        out[k] = p
    return out


def normalize_profile(profile: Dict[str, Any]) -> Dict[str, Any]:
    p = dict(profile or {})
    p.setdefault("brand_name", "BRAND")
    p.setdefault("description", "")
    p.setdefault("aliases", [])
    p.setdefault("keep_categories", [])
    p.setdefault("drop_categories", [])
    p.setdefault("sure_drop_patterns", [])
    p.setdefault("pr_reply_markers", [])
    if "brand_sure_drop" not in p or p["brand_sure_drop"] is None:
        p["brand_sure_drop"] = []
    if "homonym_noise" not in p or p["homonym_noise"] is None:
        p["homonym_noise"] = []
    if "search_noise_patterns" not in p or p["search_noise_patterns"] is None:
        p["search_noise_patterns"] = []

    def _normalize_named_categories(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        out_items = []
        for item in items or []:
            if not isinstance(item, dict):
                continue
            name = (item.get("name") or item.get(
                "category_name") or "").strip()
            if not name:
                continue
            description = (item.get("description") or "").strip()
            patterns = [str(x).strip() for x in (
                item.get("patterns") or []) if str(x).strip()]
            out_items.append(
                {
                    "name": name,
                    "description": description,
                    "patterns": patterns,
                }
            )
        return out_items

    p["keep_categories"] = _normalize_named_categories(
        p.get("keep_categories", []))
    p["drop_categories"] = _normalize_named_categories(
        p.get("drop_categories", []))
    return p


def _format_named_categories(categories: List[Dict[str, Any]]) -> str:
    lines = []
    for item in categories or []:
        name = (item.get("name") or "").strip()
        if not name:
            continue
        description = (item.get("description") or "").strip()
        if description:
            lines.append(f"- {name}: {description}")
        else:
            lines.append(f"- {name}")
    return "\n".join(lines) or "—"


def format_system_prompt(base_template: str, profile: Dict[str, Any]) -> str:
    brand_name = (profile.get("brand_name") or "BRAND").strip()
    desc = (profile.get("description") or "").strip()
    aliases = profile.get("aliases") or []
    aliases_str = ", ".join([a.strip()
                            for a in aliases if str(a).strip()]) or "—"
    keep_categories_str = _format_named_categories(
        profile.get("keep_categories", []))
    drop_categories_str = _format_named_categories(
        profile.get("drop_categories", []))
    return base_template.format(
        brand_name=brand_name,
        brand_description=desc if desc else "—",
        brand_aliases=aliases_str,
        brand_keep_categories=keep_categories_str,
        brand_drop_categories=drop_categories_str,
    ).strip()


def format_named_categories_text(categories: List[Dict[str, Any]]) -> str:
    lines = []
    for item in categories or []:
        if not isinstance(item, dict):
            continue
        name = (item.get("name") or item.get("category_name") or "").strip()
        if not name:
            continue
        description = (item.get("description") or "").strip()
        if description:
            lines.append(f"{name} - {description}")
        else:
            lines.append(name)
    return "\n".join(lines)


def parse_named_categories_text(raw_text: str) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    seen = set()
    for raw_line in (raw_text or "").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        parts = re.split(r"\s+[—-]\s+|:\s*", line, maxsplit=1)
        name = (parts[0] or "").strip()
        description = (parts[1] if len(parts) > 1 else "").strip()
        if not name or name.lower() in seen:
            continue
        seen.add(name.lower())
        out.append({"name": name, "description": description, "patterns": []})
    return out


def build_category_prompt(profile: Dict[str, Any]) -> str:
    brand_name = (profile.get("brand_name") or "BRAND").strip()
    brand_context = (profile.get("description") or "").strip()
    keep_categories = profile.get("keep_categories", []) or []

    lines = [
        f'Ты классифицируешь комментарии пользователей о бренде "{brand_name}".',
        "",
        "Выбирай строго одну категорию из списка ниже.",
        "",
        "Категории:",
    ]

    for item in keep_categories:
        name = (item.get("name") or "").strip()
        if not name:
            continue
        description = (item.get("description") or "").strip()
        if description:
            lines.append(f"- {name}: {description}")
        else:
            lines.append(f"- {name}")

    lines.extend(
        [
            "",
            f"Контекст бренда: {brand_context if brand_context else '—'}",
            "",
            "Правила:",
            "- Выбирай одну главную категорию по основной мысли комментария.",
            "- Не придумывай новые категории.",
            "- Используй только категории из списка.",
            '- Если комментарий касается нескольких тем, выбирай наиболее важную для пользователя тему.',
        ]
    )
    return "\n".join(lines).strip()


# ---------------- Helpers: OpenAI ----------------
def get_api_key() -> Optional[str]:
    secret_key = None
    try:
        secret_key = st.secrets.get("OPENAI_API_KEY", None)
    except Exception:
        secret_key = None
    return secret_key or os.getenv("OPENAI_API_KEY") or LOCAL_OPENAI_API_KEY


@st.cache_resource
def get_client(api_key: str) -> OpenAI:
    return OpenAI(api_key=api_key or "")


# ---------------- Helpers: file IO ----------------
def read_uploaded_table(uploaded_file) -> pd.DataFrame:
    name = (uploaded_file.name or "").lower()
    if name.endswith((".xlsx", ".xls")):
        return pd.read_excel(uploaded_file, engine="openpyxl")

    data = uploaded_file.getvalue()
    bio = io.BytesIO(data)
    try:
        return pd.read_csv(bio, sep=None, engine="python", encoding="utf-8-sig")
    except Exception:
        bio = io.BytesIO(data)
        try:
            return pd.read_csv(bio, sep=";", encoding="utf-8-sig")
        except Exception:
            bio = io.BytesIO(data)
            return pd.read_csv(bio, sep=",", encoding="utf-8-sig")


def df_to_download_bytes(df: pd.DataFrame, out_fmt: str) -> tuple[bytes, str]:
    if out_fmt == "xlsx":
        buff = io.BytesIO()
        with pd.ExcelWriter(buff, engine="openpyxl") as writer:
            df.to_excel(writer, index=False, sheet_name="result")
        return buff.getvalue(), "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    return df.to_csv(index=False).encode("utf-8-sig"), "text/csv"


# ---------------- Preprocessing factory (uses processing.py) ----------------
def _literal_to_pattern(s: str) -> str:
    s = (s or "").strip()
    if not s:
        return ""
    esc = re.escape(s)
    return rf"(?<!\w){esc}(?!\w)"


def build_brand_patterns(brand_name: str, aliases: List[str], extra_regex: str) -> List[str]:
    pats: List[str] = []
    for t in [brand_name] + (aliases or []):
        p = _literal_to_pattern(t)
        if p:
            pats.append(p)

    for line in (extra_regex or "").splitlines():
        line = line.strip()
        if line:
            pats.append(line)

    if not pats:
        pats = [r"(brand|brands)"]
    return pats


class PreprocessorFactory:
    def __init__(self, max_words: int = 250):
        self.max_words = max_words

    def make(self, profile: Dict[str, Any], extra_brand_patterns: str) -> proc.CommentPreprocessor:
        brand_name = profile.get("brand_name") or "BRAND"
        aliases = profile.get("aliases") or []
        pats = build_brand_patterns(brand_name, aliases, extra_brand_patterns)

        # меняем глобальную переменную processing.py "на лету"
        proc.BRAND_PATTERNS = pats

        return proc.CommentPreprocessor(
            BRAND_PATTERNS=proc.BRAND_PATTERNS,
            NOISE_PHRASES=proc.NOISE_PHRASES,
            RU_STOP=proc.RU_STOP,
            TOPIC_KEYWORDS=proc.TOPIC_KEYWORDS,
            max_len=self.max_words,  # у класса max_len трактуется как число слов
        )

    def preprocess_for_llm(self, text_rule: str, pre: proc.CommentPreprocessor) -> str:
        out = pre.preprocess(text_rule, max_len=self.max_words)
        return out if out else text_rule


# ---------------- Sentiment helpers ----------------
def _read_sentiment_meta(npz_path: str) -> dict:
    yml = Path(npz_path).with_suffix(".yaml")
    if yml.exists():
        try:
            with open(yml, "r", encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
        except Exception:
            return {}
    return {}


@st.cache_resource
def get_sentiment_model_cached(
    api_key: str,
    artifacts_npz: str,
    openai_embed_model: str,
    dimensions: int | None,
) -> SentimentModel:
    """
    Загружаем прототипы/пороги из .npz и создаём модель (inference).
    """
    client = get_client(api_key)
    embedder = OpenAIEmbedder(
        client=client, model=openai_embed_model, dimensions=dimensions)

    cfg = SentimentModelConfig(enable_llm_fallback=True)
    m = SentimentModel(embed_fn=embedder.embed_texts,
                       config=cfg, openai_api_key=api_key)
    m.load_artifacts(artifacts_npz)
    return m

# ---------------- App (class-based) ----------------


class StreamlitBrandAnalyticsApp:
    def __init__(self):
        self.brands = load_brands("brands.yaml")
        self.api_key = get_api_key()
        self.api_key_present = bool(self.api_key)

    # ---------- sidebar ----------
    def sidebar_settings(self) -> Dict[str, Any]:
        with st.sidebar:
            st.subheader("Settings")

            llm_model = st.session_state.get("llm_model", DEFAULT_LLM_MODEL)
            temperature = float(st.session_state.get("temperature", 0.0))
            truncate_chars = int(st.session_state.get("truncate_chars", 800))
            st.session_state["llm_model"] = llm_model
            st.session_state["temperature"] = temperature
            st.session_state["truncate_chars"] = truncate_chars

            brand_names = ["(manual)"] + sorted(list(self.brands.keys()))
            chosen = st.selectbox("Brand", brand_names, index=int(
                st.session_state.get("chosen_idx", 0)))
            st.session_state["chosen_idx"] = brand_names.index(chosen)

            st.markdown(
                '<div class="small-note">“manual” — вставляешь карточку бренда руками.</div>',
                unsafe_allow_html=True,
            )

            st.subheader("Sentiment")
            enable_sentiment = st.checkbox("Enable sentiment", value=bool(
                st.session_state.get("enable_sentiment", True)))
            sentiment_only_kept = st.checkbox(
                "Sentiment only for KEEP",
                value=bool(st.session_state.get("sentiment_only_kept", True)),
            )
            sentiment_artifacts = st.session_state.get(
                "sentiment_artifacts", DEFAULT_SENTIMENT_ARTIFACTS)
            st.session_state["enable_sentiment"] = bool(enable_sentiment)
            st.session_state["sentiment_only_kept"] = bool(sentiment_only_kept)
            st.session_state["sentiment_artifacts"] = sentiment_artifacts

            batch_size = int(st.session_state.get("batch_size", 6))
            max_workers = int(st.session_state.get("max_workers", 3))
            embed_model = st.session_state.get(
                "embed_model", DEFAULT_EMBED_MODEL)
            embed_batch = int(st.session_state.get("embed_batch", 128))
            st.session_state["batch_size"] = batch_size
            st.session_state["max_workers"] = max_workers
            st.session_state["embed_model"] = embed_model
            st.session_state["embed_batch"] = embed_batch

        return {
            "llm_model": llm_model,
            "temperature": float(temperature),
            "truncate_chars": int(truncate_chars),
            "chosen": chosen,
            "batch_size": int(batch_size),
            "max_workers": int(max_workers),
            "embed_model": embed_model,
            "embed_batch": int(embed_batch),
            "enable_sentiment": bool(enable_sentiment),
            "sentiment_only_kept": bool(sentiment_only_kept),
            "sentiment_artifacts": sentiment_artifacts,
        }

    # ---------- brand profile ----------
    def brand_profile_editor(self, chosen: str) -> tuple[Dict[str, Any], str]:
        overrides = st.session_state.get("brand_rule_overrides", {})
        if chosen != "(manual)" and chosen in overrides:
            profile = dict(overrides[chosen])
        elif chosen != "(manual)" and chosen in self.brands:
            profile = dict(self.brands[chosen])
        else:
            profile = {
                "brand_name": st.session_state.get("manual_brand_name", "BRAND"),
                "description": st.session_state.get("manual_description", ""),
                "aliases": st.session_state.get("manual_aliases", []),
                "keep_categories": st.session_state.get("manual_keep_categories", []),
                "drop_categories": st.session_state.get("manual_drop_categories", []),
                "sure_drop_patterns": st.session_state.get("manual_sure_drop_patterns", []),
                "pr_reply_markers": st.session_state.get("manual_pr_reply_markers", []),
                "brand_sure_drop": st.session_state.get("manual_brand_sure_drop", []),
                "homonym_noise": st.session_state.get("manual_homonym_noise", []),
                "search_noise_patterns": st.session_state.get("manual_search_noise_patterns", []),
            }

        profile = normalize_profile(profile)

        st.subheader("Brand card")
        col1, col2 = st.columns([1, 1])
        with col1:
            brand_name = st.text_input(
                "Brand name", value=profile.get("brand_name", "BRAND"))
        with col2:
            aliases_raw = st.text_input(
                "Aliases (comma-separated)",
                value=", ".join(profile.get("aliases") or []),
                placeholder="brandname, бренднейм, ...",
            )

        description = st.text_area(
            "Brand context",
            value=profile.get("description", ""),
            height=140,
            placeholder="Что это за бренд, что продаёт/делает, каналы (приложение/сайт), что считаем релевантным…",
        )

        keep_categories_raw = st.text_area(
            "Keep categories (one per line: Название - описание)",
            value=format_named_categories_text(
                profile.get("keep_categories", [])),
            height=160,
            placeholder="Покупка и ассортимент - Комментарии про товары, наличие, размеры, выбор\nПриложение и бонусы - Комментарии про приложение, сайт, оплату, бонусы",
        )

        drop_categories_raw = st.text_area(
            "Drop categories (one per line: Название - описание)",
            value=format_named_categories_text(
                profile.get("drop_categories", [])),
            height=160,
            placeholder="Вакансии и найм - Сообщения про вакансии, поиск сотрудников, HR\nЛокация и ориентир - Бренд упомянут только как гео-точка",
        )

        updated_profile = normalize_profile(
            {
                **profile,
                "brand_name": brand_name.strip() if brand_name.strip() else "BRAND",
                "aliases": [a.strip() for a in aliases_raw.split(",") if a.strip()],
                "description": description.strip(),
                "keep_categories": parse_named_categories_text(keep_categories_raw),
                "drop_categories": parse_named_categories_text(drop_categories_raw),
            }
        )

        extra_brand_patterns = ""

        with st.expander("Авто‑генерация правил (AI)", expanded=False):
            st.caption(
                "Сгенерирует regex по пользовательским drop-категориям и размеченным примерам."
            )
            ex_file = st.file_uploader(
                "Примеры (CSV/XLSX, колонки 'Текст' и 'Категория') — опционально",
                type=["csv", "xlsx", "xls"],
                key="rules_ex_file",
            )
            ex_limit = st.number_input(
                "Макс. примеров",
                min_value=5,
                max_value=100,
                value=30,
                step=5,
                key="rules_ex_limit",
            )
            gen_btn = st.button("Сгенерировать правила", key="gen_rules_btn")

            if gen_btn:
                if not self.api_key_present:
                    st.error("Нет OPENAI_API_KEY — добавь ключ в Secrets.")
                elif not updated_profile.get("drop_categories"):
                    st.error("Сначала задай хотя бы одну drop-категорию.")
                else:
                    examples: List[Dict[str, str]] = []
                    examples_invalid = False
                    if ex_file is not None:
                        try:
                            df_ex = read_uploaded_table(ex_file)
                            if "Текст" not in df_ex.columns or "Категория" not in df_ex.columns:
                                st.error(
                                    "Файл примеров должен содержать колонки 'Текст' и 'Категория'.")
                                examples_invalid = True
                                df_ex = None
                            if df_ex is not None:
                                allowed = {item["name"] for item in updated_profile.get(
                                    "drop_categories", [])}
                                subset = df_ex[["Текст", "Категория"]].copy()
                                subset["Текст"] = subset["Текст"].astype(
                                    str).fillna("").str.strip()
                                subset["Категория"] = subset["Категория"].astype(
                                    str).fillna("").str.strip()
                                subset = subset[(subset["Текст"] != "") & (
                                    subset["Категория"] != "")]
                                unknown = sorted(
                                    {x for x in subset["Категория"].tolist() if x not in allowed})
                                if unknown:
                                    st.error(
                                        "В файле примеров есть категории, которых нет в списке drop-категорий: "
                                        + ", ".join(unknown)
                                    )
                                    examples_invalid = True
                                    subset = subset.iloc[0:0]
                                for _, row in subset.head(int(ex_limit)).iterrows():
                                    examples.append(
                                        {
                                            "text": str(row["Текст"])[:400],
                                            "category": str(row["Категория"]),
                                        }
                                    )
                        except Exception as e:
                            st.warning("Не удалось прочитать примеры.")
                            st.exception(e)
                            examples_invalid = True

                    if not examples_invalid:
                        client = get_client(self.api_key or "")
                        try:
                            parsed = generate_rules(
                                updated_profile,
                                updated_profile.get("drop_categories", []),
                                examples,
                                client=client,
                                model=st.session_state.get(
                                    "llm_model", DEFAULT_LLM_MODEL),
                                temperature=0.2,
                            )
                            patterns_by_name = {
                                item.category_name: item.patterns for item in parsed.rules
                            }
                            updated_profile["drop_categories"] = [
                                {
                                    "name": item["name"],
                                    "description": item.get("description", ""),
                                    "patterns": patterns_by_name.get(item["name"], []),
                                }
                                for item in updated_profile.get("drop_categories", [])
                            ]

                            # persist for session
                            overrides = dict(st.session_state.get(
                                "brand_rule_overrides", {}))
                            if chosen != "(manual)":
                                overrides[chosen] = dict(updated_profile)
                                st.session_state["brand_rule_overrides"] = overrides
                            else:
                                st.session_state["manual_drop_categories"] = updated_profile["drop_categories"]

                            st.success(
                                f"Правила сгенерированы для {len(updated_profile['drop_categories'])} категорий."
                            )
                        except Exception as e:
                            st.error("Не удалось сгенерировать правила.")
                            st.exception(e)

        profile = updated_profile

        if chosen == "(manual)":
            st.session_state["manual_brand_name"] = profile["brand_name"]
            st.session_state["manual_aliases"] = profile["aliases"]
            st.session_state["manual_description"] = profile["description"]
            st.session_state["manual_keep_categories"] = profile["keep_categories"]
            st.session_state["manual_drop_categories"] = profile["drop_categories"]
            st.session_state["manual_sure_drop_patterns"] = profile["sure_drop_patterns"]
            st.session_state["manual_pr_reply_markers"] = profile["pr_reply_markers"]
            st.session_state["manual_brand_sure_drop"] = profile["brand_sure_drop"]
            st.session_state["manual_homonym_noise"] = profile.get(
                "homonym_noise", [])
            st.session_state["manual_search_noise_patterns"] = profile.get(
                "search_noise_patterns", [])

        return profile, extra_brand_patterns

    # ---------- system prompt ----------
    def system_prompt_section(self, profile: Dict[str, Any]) -> str:
        return format_system_prompt(BASE_SYSTEM_TEMPLATE, profile)

    # ---------- readiness ----------
    def ensure_ready(self):
        if _IMPORT_ERR is not None:
            st.error("Не найдены файлы llm_model.py и/или filter_service.py.")
            st.code(str(_IMPORT_ERR), language="text")
            st.stop()

        if not self.api_key_present:
            st.warning(
                "Не найден OPENAI_API_KEY. Добавь ключ в Secrets или env.")
            # UI ок, но запуск блокируем по кнопке

    # ---------- sentiment init ----------
    def _build_sentiment_service(
        self,
        *,
        preproc_factory: PreprocessorFactory,
        pre: proc.CommentPreprocessor,
        artifacts_npz: str,
        openai_embed_model: str,
        embed_batch_size: int,
    ) -> Optional[SentimentService]:
        if not artifacts_npz or not Path(artifacts_npz).exists():
            return None

        meta = _read_sentiment_meta(artifacts_npz)
        dimensions = meta.get("dimensions", None)
        if isinstance(dimensions, str) and dimensions.isdigit():
            dimensions = int(dimensions)
        if isinstance(dimensions, (int, float)):
            dimensions = int(dimensions)
        else:
            dimensions = None

        # если meta содержит embed_model и пользователь не менял — можно подхватить
        # но в UI мы явно выбираем модель, так что используем openai_embed_model

        model = get_sentiment_model_cached(
            api_key=self.api_key or "",
            artifacts_npz=artifacts_npz,
            openai_embed_model=openai_embed_model,
            dimensions=dimensions,
        )

        def preprocess_fn(t: str) -> str:
            return preproc_factory.preprocess_for_llm(t, pre)

        return SentimentService(model=model, preprocess_fn=preprocess_fn, embed_batch_size=int(embed_batch_size))

    # ---------- render: file (relevance + sentiment) ----------
    def render_file(
        self,
        *,
        profile: Dict[str, Any],
        final_system: str,
        extra_brand_patterns: str,
        model: str,
        temperature: float,
        truncate_chars: int,
        batch_size: int,
        max_workers: int,
        enable_sentiment: bool,
        sentiment_only_kept: bool,
        sentiment_artifacts: str,
        embed_model: str,
        embed_batch: int,
    ):
        st.subheader("File (CSV/XLSX)")
        uploaded = st.file_uploader(
            "CSV/Excel with required column: Текст", type=["csv", "xlsx", "xls"], key="file_upl")

        run_file = st.button("🚀 Process file", type="primary",
                             use_container_width=True, key="run_file_btn")

        if not run_file:
            return

        if not self.api_key_present:
            st.error("Нет OPENAI_API_KEY — добавь ключ в Secrets.")
            st.stop()

        if uploaded is None:
            st.error("Upload CSV/XLSX first.")
            st.stop()

        try:
            df_in = read_uploaded_table(uploaded)
        except Exception as e:
            st.error("Не смог прочитать файл. Проверь формат CSV/XLSX.")
            st.exception(e)
            st.stop()

        if "Текст" not in df_in.columns:
            st.error(
                f"В файле нет столбца 'Текст'. Есть: {list(df_in.columns)}")
            st.stop()

        texts = df_in["Текст"].astype(str).fillna("").tolist()

        preproc_factory = PreprocessorFactory(max_words=250)
        pre = preproc_factory.make(profile, extra_brand_patterns)

        def preprocess_fn(text_rule: str) -> str:
            return preproc_factory.preprocess_for_llm(text_rule, pre)

        client = get_client(self.api_key or "")
        llm = OpenAIRelevanceBatchModel(client=client, default_model=model)
        service = RelevanceFilterService(llm=llm)
        progress_bar = st.progress(0.0, text="Preparing Relevance…")

        def _set_pipeline_progress(progress_value: float, text: str) -> None:
            progress_bar.progress(max(0.0, min(1.0, float(progress_value))), text=text)

        def _on_relevance_progress(done: int, total: int, stage: str) -> None:
            total_safe = max(1, int(total))
            progress = 0.5 if enable_sentiment else 1.0
            current = progress * (int(done) / total_safe)
            _set_pipeline_progress(
                current,
                f"Relevance: {int(done)}/{int(total)} messages",
            )

        # ---- relevance ----
        with st.spinner("Relevance (RULE + LLM batches)…"):
            actions, stats = service.classify_many_parallel(
                texts=texts,
                profile=profile,
                system_prompt=final_system,
                preprocess_fn=preprocess_fn,
                batch_size=batch_size,
                max_workers=max_workers,
                truncate_chars=truncate_chars,
                model=model,
                temperature=temperature,
                progress_callback=_on_relevance_progress,
            )

        if not enable_sentiment:
            _set_pipeline_progress(1.0, f"Relevance complete: {len(texts)}/{len(texts)} messages")

        df_out = df_in.copy()
        df_out["is_drop"] = ["Yes" if a == "drop" else "No" for a in actions]

        # ---- sentiment (optional) ----
        if enable_sentiment:
            sent_service = self._build_sentiment_service(
                preproc_factory=preproc_factory,
                pre=pre,
                artifacts_npz=sentiment_artifacts,
                openai_embed_model=embed_model,
                embed_batch_size=embed_batch,
            )
            if sent_service is None:
                st.warning(
                    "Sentiment: assets not found — sentiment will be skipped.")
                _set_pipeline_progress(1.0, "Sentiment skipped: assets not found")
            else:
                with st.spinner("Sentiment inference…"):
                    sentiment_total = 0

                    def _on_sentiment_progress(done: int, total: int, stage: str) -> None:
                        total_safe = max(1, int(total))
                        current = 0.5 + 0.5 * (int(done) / total_safe)
                        _set_pipeline_progress(
                            current,
                            f"Sentiment: {int(done)}/{int(total)} messages",
                        )

                    if sentiment_only_kept:
                        kept_mask = df_out["is_drop"].astype(
                            str).str.lower().eq("no").tolist()
                        kept_texts = [t for t, k in zip(texts, kept_mask) if k]
                        sentiment_total = len(kept_texts)
                        if sentiment_total == 0:
                            _set_pipeline_progress(1.0, "Sentiment: 0/0 messages")
                            labels, sources = [], []
                        else:
                            labels, sources = sent_service.predict_many(
                                kept_texts,
                                progress_callback=_on_sentiment_progress,
                            )

                        sent_col = []
                        src_col = []
                        it = iter(zip(labels, sources))
                        for k in kept_mask:
                            if k:
                                lab, src = next(it)
                                sent_col.append(lab)
                                src_col.append(src)
                            else:
                                sent_col.append(None)
                                src_col.append("skip_drop")
                        df_out["sentiment"] = sent_col
                        df_out["sentiment_source"] = src_col
                    else:
                        sentiment_total = len(texts)
                        labels, sources = sent_service.predict_many(
                            texts,
                            progress_callback=_on_sentiment_progress,
                        )
                        df_out["sentiment"] = labels
                        df_out["sentiment_source"] = sources

                _set_pipeline_progress(
                    1.0,
                    f"Sentiment complete: {sentiment_total}/{sentiment_total} messages",
                )

        # save for categories step
        st.session_state["last_filtered_df"] = df_out.copy()
        st.session_state["last_filtered_name"] = f"filtered_{Path(uploaded.name).stem}"

        # ---- UI ----
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Done")

        n = int(stats.get("n", len(texts)))
        rule_drops = int(stats.get("rule_drops", 0))
        llm_calls = int(stats.get("llm_calls", 0))
        total_s = float(stats.get("total_s", 0.0))
        cps = float(stats.get("comments_per_s", 0.0) or 0.0)
        drops = int((df_out["is_drop"].astype(str).str.lower() == "yes").sum())
        keeps = int(n - drops)

        badges = [
            f'<span class="badge badge--red">DROP: {drops}</span>',
            f'<span class="badge badge--green">KEEP: {keeps}</span>',
            f'<span class="badge badge--blue">LLM calls: {llm_calls}</span>',
        ]
        if enable_sentiment and "sentiment" in df_out.columns:
            pos = int((df_out["sentiment"] == "positive").sum())
            neg = int((df_out["sentiment"] == "negative").sum())
            neu = int((df_out["sentiment"] == "neutral").sum())
            badges += [
                f'<span class="badge badge--green">POS: {pos}</span>',
                f'<span class="badge badge--red">NEG: {neg}</span>',
                f'<span class="badge badge--gray">NEU: {neu}</span>',
            ]

        st.markdown(
            f'<div class="badge-row">{"".join(badges)}</div>', unsafe_allow_html=True)
        st.caption(
            f"Rows: {n} • RULE drops: {rule_drops} • Total: {total_s:.2f}s • comments/s: {cps:.2f}"
        )

        st.dataframe(df_out.head(30), use_container_width=True)

        name = (uploaded.name or "result").lower()
        out_fmt = "xlsx" if name.endswith((".xlsx", ".xls")) else "csv"
        out_bytes, mime = df_to_download_bytes(df_out, out_fmt=out_fmt)
        out_name = f"filtered_{Path(uploaded.name).stem}.{out_fmt}"

        st.download_button(
            "⬇️ Download result",
            data=out_bytes,
            file_name=out_name,
            mime=mime,
            use_container_width=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

    # ---------- render: categories ----------
    def render_categories_file(
        self,
        *,
        profile: Dict[str, Any],
        extra_brand_patterns: str,
        llm_model: str,
        embed_model: str,
        embed_batch: int,
        truncate_chars: int,
        batch_size: int,
        max_workers: int,
    ):
        st.subheader("Category tagging (file)")

        if _CAT_IMPORT_ERR is not None or CategoryTagger is None:
            st.error("Не найдены category_model.py / category_service.py")
            st.code(str(_CAT_IMPORT_ERR), language="text")
            st.stop()

        # input: last filtered df or uploaded file
        has_last = st.session_state.get("last_filtered_df") is not None
        use_last = st.checkbox(
            "Use last Relevance+Sentiment output",
            value=bool(has_last),
            disabled=not has_last,
        )

        df_in = None
        uploaded = None

        if use_last and has_last:
            df_in = st.session_state["last_filtered_df"].copy()
            st.caption("Input taken from previous step (Relevance+Sentiment).")
        else:
            uploaded = st.file_uploader(
                "Input CSV/XLSX with column 'Текст' (optional: is_drop)",
                type=["csv", "xlsx", "xls"],
                key="cat_input_file",
            )
            if uploaded is not None:
                df_in = read_uploaded_table(uploaded)

        ref_uploaded = st.file_uploader(
            "Labeled dataset (optional for RAG) — columns: 'Текст' and 'Категория'",
            type=["csv", "xlsx", "xls"],
            key="cat_ref_file",
        )

        run_btn = st.button("🚀 Tag categories", type="primary",
                            use_container_width=True, key="run_cat_btn")

        if not run_btn:
            return

        if not self.api_key_present:
            st.error("Нет OPENAI_API_KEY — добавь ключ в Secrets.")
            st.stop()

        if df_in is None:
            st.error("Need input file (or enable 'Use last output').")
            st.stop()

        if "Текст" not in df_in.columns:
            st.error("Input must contain column 'Текст'.")
            st.stop()

        allowed_categories = [
            item["name"] for item in (profile.get("keep_categories") or []) if (item.get("name") or "").strip()
        ]
        if not allowed_categories:
            st.error("Сначала задай хотя бы одну keep-категорию в карточке бренда.")
            st.stop()
        user_prompt = build_category_prompt(profile)

        preproc_factory = PreprocessorFactory(max_words=250)
        pre = preproc_factory.make(profile, extra_brand_patterns)

        def preprocess_fn(text_rule: str) -> str:
            return preproc_factory.preprocess_for_llm(text_rule, pre)

        client = get_client(self.api_key or "")
        embedder = OpenAIEmbedder(
            client=client, model=embed_model, dimensions=None)

        tagger = CategoryTagger(
            client=client,
            embedder=embedder,
            llm_model=llm_model,
            max_output_tokens=1200,
            temperature=0.0,
        )
        svc = CategoryTaggingService(tagger=tagger)
        progress_bar = st.progress(0.0, text="Preparing Category tagging…")

        def _on_category_progress(done: int, total: int, stage: str) -> None:
            total_safe = max(1, int(total))
            progress_bar.progress(
                max(0.0, min(1.0, int(done) / total_safe)),
                text=f"Category tagging: {int(done)}/{int(total)} messages",
            )

        # build/cache ref index (optional)
        ref_index: Optional[CategoryIndex] = None
        if ref_uploaded is not None:
            try:
                ref_bytes = ref_uploaded.getvalue()
                cache_key = hashlib.md5(
                    ref_bytes).hexdigest() + f"::{embed_model}"

                cache: Dict[str, Any] = st.session_state.get(
                    "cat_ref_cache", {})
                if cache_key in cache:
                    ref_index = cache[cache_key]
                else:
                    df_ref = read_uploaded_table(ref_uploaded)
                    if "Текст" not in df_ref.columns or "Категория" not in df_ref.columns:
                        st.error(
                            "Labeled dataset must contain columns 'Текст' and 'Категория'.")
                        st.stop()
                    ref_categories = df_ref["Категория"].astype(
                        str).fillna("").str.strip()
                    unknown_cats = sorted(
                        {x for x in ref_categories.tolist() if x and x not in allowed_categories})
                    if unknown_cats:
                        st.error(
                            "Labeled dataset contains categories outside Keep categories: "
                            + ", ".join(unknown_cats)
                        )
                        st.stop()

                    ref_texts = [preprocess_fn(x) for x in df_ref["Текст"].astype(
                        str).fillna("").tolist()]
                    ref_cats = ref_categories.tolist()

                    with st.spinner("Building embedding index for labeled dataset…"):
                        ref_index = tagger.build_index(
                            ref_texts=ref_texts, ref_cats=ref_cats, embed_batch_size=int(embed_batch))

                    cache[cache_key] = ref_index
                    st.session_state["cat_ref_cache"] = cache

                st.caption(
                    f"RAG ON ✅ (unique categories: {len(ref_index.categories)})")
            except Exception as e:
                st.error("Failed to build RAG index.")
                st.exception(e)
                st.stop()
        else:
            st.caption("RAG OFF → LLM-only ✅")

        with st.spinner("Tagging categories…"):
            df_out, meta = svc.run(
                df_in=df_in,
                text_col="Текст",
                user_prompt=user_prompt,
                allowed_categories=allowed_categories,
                preprocess_fn=preprocess_fn,
                ref_index=ref_index,
                is_drop_col="is_drop" if "is_drop" in df_in.columns else None,
                top_k=5,
                llm_batch_size=int(batch_size),
                max_workers=int(max_workers),
                truncate_chars=int(truncate_chars),
                embed_batch_size=int(embed_batch),
                progress_callback=_on_category_progress,
            )

        progress_bar.progress(1.0, text=f"Category tagging complete: {len(df_in)}/{len(df_in)} messages")

        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.header("Result (Categories)")

        badges = [
            f'<span class="badge badge--blue">rows: {meta.get("rows")}</span>',
            f'<span class="badge badge--blue">classified: {meta.get("classified")}</span>',
            f'<span class="badge badge--gray">skipped_drop: {meta.get("skipped_drop")}</span>',
            f'<span class="badge badge--blue">mode: {"RAG" if meta.get("use_rag") else "LLM"}</span>',
        ]
        st.markdown(
            f'<div class="badge-row">{"".join(badges)}</div>', unsafe_allow_html=True)
        st.caption(meta)

        st.dataframe(df_out.head(30), use_container_width=True)

        out_bytes, mime = df_to_download_bytes(df_out, out_fmt="xlsx")
        out_name = "categorized.xlsx"
        st.download_button(
            "⬇️ Download (xlsx)",
            data=out_bytes,
            file_name=out_name,
            mime=mime,
            use_container_width=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

    # ---------- main ----------
    def run(self):
        st.title("Brand Analytics")
        st.caption(
            "Relevance filtering + Sentiment + Category tagging (RAG optional).")

        self.ensure_ready()

        settings = self.sidebar_settings()
        profile, extra_brand_patterns = self.brand_profile_editor(
            settings["chosen"])
        final_system = self.system_prompt_section(profile)

        tool = st.radio(
            "Tool",
            ["Relevance + Sentiment", "Category tagging"],
            horizontal=True,
        )

        if tool == "Relevance + Sentiment":
            self.render_file(
                profile=profile,
                final_system=final_system,
                extra_brand_patterns=extra_brand_patterns,
                model=settings["llm_model"],
                temperature=settings["temperature"],
                truncate_chars=settings["truncate_chars"],
                batch_size=settings["batch_size"],
                max_workers=settings["max_workers"],
                enable_sentiment=settings["enable_sentiment"],
                sentiment_only_kept=settings["sentiment_only_kept"],
                sentiment_artifacts=settings["sentiment_artifacts"],
                embed_model=settings["embed_model"],
                embed_batch=settings["embed_batch"],
            )

        else:
            self.render_categories_file(
                profile=profile,
                extra_brand_patterns=extra_brand_patterns,
                llm_model=settings["llm_model"],
                embed_model=settings["embed_model"],
                embed_batch=settings["embed_batch"],
                truncate_chars=settings["truncate_chars"],
                batch_size=settings["batch_size"],
                max_workers=settings["max_workers"],
            )


StreamlitBrandAnalyticsApp().run()
