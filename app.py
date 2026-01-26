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

DEFAULT_LLM_MODEL = "gpt-4.1-mini"
DEFAULT_EMBED_MODEL = "text-embedding-3-small"
DEFAULT_SENTIMENT_ARTIFACTS = "sentiment_assets/sentiment_openai.npz"

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
        p.setdefault("sure_drop_patterns", [])
        p.setdefault("pr_reply_markers", [])
        # совместимость с RuleEngine из filter_service.py
        p.setdefault("brand_sure_drop", p.get("sure_drop_patterns", []))
        p.setdefault("homonym_noise", [])
        out[k] = p
    return out


def normalize_profile(profile: Dict[str, Any]) -> Dict[str, Any]:
    p = dict(profile or {})
    p.setdefault("brand_name", "BRAND")
    p.setdefault("description", "")
    p.setdefault("aliases", [])
    p.setdefault("sure_drop_patterns", [])
    p.setdefault("pr_reply_markers", [])
    if "brand_sure_drop" not in p or p["brand_sure_drop"] is None:
        p["brand_sure_drop"] = p.get("sure_drop_patterns", [])
    if "homonym_noise" not in p or p["homonym_noise"] is None:
        p["homonym_noise"] = []
    return p


def format_system_prompt(base_template: str, profile: Dict[str, Any]) -> str:
    brand_name = (profile.get("brand_name") or "BRAND").strip()
    desc = (profile.get("description") or "").strip()
    aliases = profile.get("aliases") or []
    aliases_str = ", ".join([a.strip()
                            for a in aliases if str(a).strip()]) or "—"
    return base_template.format(
        brand_name=brand_name,
        brand_description=desc if desc else "—",
        brand_aliases=aliases_str,
    ).strip()


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

    cfg = SentimentModelConfig(enable_llm_fallback=False)
    m = SentimentModel(embed_fn=embedder.embed_texts,
                       config=cfg, openai_api_key=api_key)
    m.load_artifacts(artifacts_npz)
    return m


def sentiment_badge(label: str) -> str:
    l = (label or "").strip().lower()
    if l == "positive":
        return '<span class="badge badge--green">SENTIMENT: POSITIVE</span>'
    if l == "negative":
        return '<span class="badge badge--red">SENTIMENT: NEGATIVE</span>'
    return '<span class="badge badge--gray">SENTIMENT: NEUTRAL</span>'


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

            llm_model = st.text_input(
                "LLM model", value=st.session_state.get("llm_model", DEFAULT_LLM_MODEL))
            temperature = st.slider("Temperature", 0.0, 1.0, float(
                st.session_state.get("temperature", 0.0)), 0.1)

            truncate_chars = st.number_input(
                "Truncate chars",
                min_value=100,
                max_value=5000,
                value=int(st.session_state.get("truncate_chars", 800)),
                step=50,
            )

            st.session_state["llm_model"] = llm_model
            st.session_state["temperature"] = float(temperature)
            st.session_state["truncate_chars"] = int(truncate_chars)

            brand_names = ["(manual)"] + sorted(list(self.brands.keys()))
            chosen = st.selectbox("Brand", brand_names, index=int(
                st.session_state.get("chosen_idx", 0)))
            st.session_state["chosen_idx"] = brand_names.index(chosen)

            st.markdown(
                '<div class="small-note">“manual” — вставляешь карточку бренда руками.</div>',
                unsafe_allow_html=True,
            )

            st.subheader("File processing")
            batch_size = st.number_input("Batch size", 1, 50, int(
                st.session_state.get("batch_size", 6)), 1)
            max_workers = st.number_input("Max workers", 1, 20, int(
                st.session_state.get("max_workers", 3)), 1)
            st.session_state["batch_size"] = int(batch_size)
            st.session_state["max_workers"] = int(max_workers)

            st.subheader("Embeddings")
            embed_model = st.text_input(
                "OpenAI embedding model",
                value=st.session_state.get("embed_model", DEFAULT_EMBED_MODEL),
            )
            embed_batch = st.number_input("Embed batch size", 8, 2048, int(
                st.session_state.get("embed_batch", 128)), 8)

            st.session_state["embed_model"] = embed_model
            st.session_state["embed_batch"] = int(embed_batch)

            st.subheader("Sentiment")
            enable_sentiment = st.checkbox("Enable sentiment", value=bool(
                st.session_state.get("enable_sentiment", True)))
            sentiment_only_kept = st.checkbox(
                "Sentiment only for KEEP",
                value=bool(st.session_state.get("sentiment_only_kept", True)),
            )
            sentiment_artifacts = st.text_input(
                "Sentiment artifacts (.npz)",
                value=st.session_state.get(
                    "sentiment_artifacts", DEFAULT_SENTIMENT_ARTIFACTS),
            )

            st.session_state["enable_sentiment"] = bool(enable_sentiment)
            st.session_state["sentiment_only_kept"] = bool(sentiment_only_kept)
            st.session_state["sentiment_artifacts"] = sentiment_artifacts

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
        if chosen != "(manual)" and chosen in self.brands:
            profile = dict(self.brands[chosen])
        else:
            profile = {
                "brand_name": st.session_state.get("manual_brand_name", "BRAND"),
                "description": st.session_state.get("manual_description", ""),
                "aliases": st.session_state.get("manual_aliases", []),
                "sure_drop_patterns": st.session_state.get("manual_sure_drop_patterns", []),
                "pr_reply_markers": st.session_state.get("manual_pr_reply_markers", []),
                "brand_sure_drop": st.session_state.get("manual_brand_sure_drop", []),
                "homonym_noise": st.session_state.get("manual_homonym_noise", []),
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
                placeholder="familia, фамилия, ...",
            )

        description = st.text_area(
            "Brand context",
            value=profile.get("description", ""),
            height=140,
            placeholder="Что это за бренд, что продаёт/делает, каналы (приложение/сайт), что считаем релевантным…",
        )

        # ВАЖНО: перенос “бренд-паттернов” вниз (ниже) — как ты просил
        with st.expander("Правила «точно drop» (бренд-специфичные регэкспы)"):
            sure_drop_text = st.text_area(
                "По одному паттерну в строке",
                value="\n".join(profile.get("sure_drop_patterns") or []),
                height=140,
                placeholder=r"(?i)\bsagrada\s+familia\b",
            )

        with st.expander("Бренд-паттерны для предобработки (regex)", expanded=False):
            extra_brand_patterns = st.text_area(
                "Каждая строка — отдельный regex. Добавится к brand_name и aliases.",
                value=st.session_state.get("extra_brand_patterns", ""),
                height=120,
                key="extra_brand_patterns",
                placeholder=r"например:\n(?<!\w)familia(?!\w)\n(?<!\w)фамилия(?!\w)\n",
            )

        with st.expander("Маркеры PR/официальных ответов (регэкспы)"):
            pr_text = st.text_area(
                "По одному паттерну в строке",
                value="\n".join(profile.get("pr_reply_markers") or []),
                height=120,
                placeholder=r"(?i)^здравствуйте",
            )

        profile["brand_name"] = brand_name.strip(
        ) if brand_name.strip() else "BRAND"
        profile["aliases"] = [a.strip()
                              for a in aliases_raw.split(",") if a.strip()]
        profile["description"] = description.strip()
        profile["sure_drop_patterns"] = [line.strip()
                                         for line in sure_drop_text.splitlines() if line.strip()]
        profile["pr_reply_markers"] = [line.strip()
                                       for line in pr_text.splitlines() if line.strip()]

        # для сервис-класса: прокинем совместимые ключи
        profile["brand_sure_drop"] = profile["sure_drop_patterns"]
        profile.setdefault("homonym_noise", [])

        if chosen == "(manual)":
            st.session_state["manual_brand_name"] = profile["brand_name"]
            st.session_state["manual_aliases"] = profile["aliases"]
            st.session_state["manual_description"] = profile["description"]
            st.session_state["manual_sure_drop_patterns"] = profile["sure_drop_patterns"]
            st.session_state["manual_pr_reply_markers"] = profile["pr_reply_markers"]
            st.session_state["manual_brand_sure_drop"] = profile["brand_sure_drop"]
            st.session_state["manual_homonym_noise"] = profile.get(
                "homonym_noise", [])

        return profile, extra_brand_patterns

    # ---------- system prompt ----------
    def system_prompt_section(self, profile: Dict[str, Any]) -> str:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("System prompt template")

        base_template = st.text_area(
            "Template (editable)",
            value=st.session_state.get("base_template", BASE_SYSTEM_TEMPLATE),
            height=220,
        )
        st.session_state["base_template"] = base_template

        final_system = format_system_prompt(base_template, profile)

        with st.expander("Preview: final system prompt"):
            st.code(final_system, language="text")

        st.markdown("</div>", unsafe_allow_html=True)
        return final_system

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

    # ---------- render: single ----------
    def render_single(
        self,
        *,
        profile: Dict[str, Any],
        final_system: str,
        extra_brand_patterns: str,
        model: str,
        temperature: float,
        truncate_chars: int,
        enable_sentiment: bool,
        sentiment_only_kept: bool,
        sentiment_artifacts: str,
        embed_model: str,
        embed_batch: int,
    ):
        st.subheader("Single comment")

        single_text = st.text_area(
            "Paste comment text",
            height=180,
            placeholder="One comment here…",
            key="single_text",
        )

        run_one = st.button("🚀 Run", type="primary",
                            use_container_width=True, key="run_one_btn")

        if not run_one:
            return

        if not self.api_key_present:
            st.error("Нет OPENAI_API_KEY — добавь ключ в Secrets.")
            st.stop()

        preproc_factory = PreprocessorFactory(max_words=250)
        pre = preproc_factory.make(profile, extra_brand_patterns)

        def preprocess_fn(text_rule: str) -> str:
            return preproc_factory.preprocess_for_llm(text_rule, pre)

        client = get_client(self.api_key or "")
        llm = OpenAIRelevanceBatchModel(client=client, default_model=model)
        service = RelevanceFilterService(llm=llm)

        # ---- relevance ----
        with st.spinner("Relevance…"):
            t0 = time.perf_counter()
            res = service.classify_one(
                raw_text=single_text,
                profile=profile,
                system_prompt=final_system,
                preprocess_fn=preprocess_fn,
                truncate_chars=truncate_chars,
                model=model,
                temperature=temperature,
            )
            dt_total = time.perf_counter() - t0

        action = res.get("action", "keep")
        is_drop = action == "drop"
        source = res.get("source", "llm")

        # ---- sentiment (optional) ----
        sent_label = None
        sent_meta = None

        if enable_sentiment:
            if sentiment_only_kept and is_drop:
                sent_label = None
                sent_meta = {"skipped": True, "reason": "is_drop=Yes"}
            else:
                sent_service = self._build_sentiment_service(
                    preproc_factory=preproc_factory,
                    pre=pre,
                    artifacts_npz=sentiment_artifacts,
                    openai_embed_model=embed_model,
                    embed_batch_size=embed_batch,
                )
                if sent_service is None:
                    sent_meta = {"skipped": True,
                                 "reason": "sentiment assets not found"}
                else:
                    with st.spinner("Sentiment…"):
                        sres = sent_service.predict_one(single_text)
                    sent_label = sres.get("label")
                    sent_meta = sres

        # ---- UI ----
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.header("Result")

        badges: List[str] = []
        if is_drop:
            badges.append(
                '<span class="badge badge--red">RELEVANCE: DROP</span>')
        else:
            badges.append(
                '<span class="badge badge--green">RELEVANCE: KEEP</span>')

        if source == "rule":
            badges.append(
                '<span class="badge badge--blue">SOURCE: RULE</span>')
        else:
            badges.append('<span class="badge badge--blue">SOURCE: LLM</span>')

        if enable_sentiment:
            if sent_meta and sent_meta.get("skipped"):
                badges.append(
                    '<span class="badge badge--gray">SENTIMENT: SKIPPED</span>')
            elif sent_label:
                badges.append(sentiment_badge(sent_label))

        st.markdown(
            f'<div class="badge-row">{"".join(badges)}</div>', unsafe_allow_html=True)

        if source == "rule":
            st.caption(
                f"Pre-LLM rule: {res.get('rule', {}).get('rule_code', 'rule')}")
        else:
            st.caption(
                f"Latency: {res.get('latency_s', 0.0):.3f}s • total: {dt_total:.3f}s")

        with st.expander("Details (JSON)"):
            payload = {"relevance": {"results": [
                {"global_idx": 0, "action": action}]}}
            if enable_sentiment:
                payload["sentiment"] = sent_meta
            st.json(payload)

        st.markdown("</div>", unsafe_allow_html=True)

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
            )

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
            else:
                with st.spinner("Sentiment inference…"):
                    if sentiment_only_kept:
                        kept_mask = df_out["is_drop"].astype(
                            str).str.lower().eq("no").tolist()
                        kept_texts = [t for t, k in zip(texts, kept_mask) if k]
                        labels, sources = sent_service.predict_many(kept_texts)

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
                        labels, sources = sent_service.predict_many(texts)
                        df_out["sentiment"] = labels
                        df_out["sentiment_source"] = sources

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

        user_prompt = st.text_area(
            "Category definitions prompt (required)",
            height=220,
            placeholder="Define categories, decision rules, edge cases, what to do if unclear…",
            key="cat_user_prompt",
        )

        col1, col2, col3 = st.columns(3)
        with col1:
            top_k = st.number_input(
                "top_k (RAG)", min_value=1, max_value=20, value=5, step=1)
        with col2:
            llm_batch_size = st.number_input(
                "LLM batch size", min_value=1, max_value=30, value=int(batch_size), step=1)
        with col3:
            mw = st.number_input("max_workers", min_value=1,
                                 max_value=12, value=int(max_workers), step=1)

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

        if not user_prompt.strip():
            st.error(
                "Category prompt is required. Please write definitions and rules.")
            st.stop()

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

                    ref_texts = [preprocess_fn(x) for x in df_ref["Текст"].astype(
                        str).fillna("").tolist()]
                    ref_cats = df_ref["Категория"].astype(
                        str).fillna("").tolist()

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
                preprocess_fn=preprocess_fn,
                ref_index=ref_index,
                is_drop_col="is_drop" if "is_drop" in df_in.columns else None,
                top_k=int(top_k),
                llm_batch_size=int(llm_batch_size),
                max_workers=int(mw),
                truncate_chars=int(truncate_chars),
                embed_batch_size=int(embed_batch),
            )

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
            mode = st.radio("Mode", ["Single comment",
                            "File (CSV/XLSX)"], horizontal=True)

            if mode == "File (CSV/XLSX)":
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
                self.render_single(
                    profile=profile,
                    final_system=final_system,
                    extra_brand_patterns=extra_brand_patterns,
                    model=settings["llm_model"],
                    temperature=settings["temperature"],
                    truncate_chars=settings["truncate_chars"],
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
