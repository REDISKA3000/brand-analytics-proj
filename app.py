# app.py
import os
import io
import re
import time
from pathlib import Path
from typing import Any, Dict, Optional, List

import pandas as pd
import streamlit as st
import yaml
from openai import OpenAI

import processing as proc  # processing.py рядом с app.py

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
  margin-bottom: 14px;
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
        # старые ключи
        p.setdefault("sure_drop_patterns", [])
        p.setdefault("pr_reply_markers", [])
        # новые ключи (для сервис-класса)
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

    # совместимость с RuleEngine в filter_service.py
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
def get_client() -> OpenAI:
    api_key = get_api_key()
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


# ---------------- App (class-based) ----------------
class StreamlitRelevanceApp:
    def __init__(self):
        self.brands = load_brands("brands.yaml")
        self.api_key_present = bool(get_api_key())

    def sidebar_settings(self) -> Dict[str, Any]:
        with st.sidebar:
            st.subheader("Настройки")
            model = st.text_input(
                "Model", value=st.session_state.get("model", DEFAULT_MODEL))
            temperature = st.slider("Temperature", 0.0, 1.0, float(
                st.session_state.get("temperature", 0.0)), 0.1)
            truncate_chars = st.number_input("Truncate chars", min_value=100, max_value=5000, value=int(
                st.session_state.get("truncate_chars", 800)), step=50)

            st.session_state["model"] = model
            st.session_state["temperature"] = temperature
            st.session_state["truncate_chars"] = truncate_chars

            brand_names = ["(manual)"] + sorted(list(self.brands.keys()))
            chosen = st.selectbox("Компания", brand_names, index=int(
                st.session_state.get("chosen_idx", 0)))
            st.session_state["chosen_idx"] = brand_names.index(chosen)

            st.markdown(
                '<div class="small-note">Под «manual» можно вставить карточку бренда руками.</div>', unsafe_allow_html=True)

            st.subheader("Параллельная обработка (для файлов)")
            batch_size = st.number_input("Batch size", 1, 50, int(
                st.session_state.get("batch_size", 6)), 1)
            max_workers = st.number_input("Max workers", 1, 20, int(
                st.session_state.get("max_workers", 3)), 1)
            st.session_state["batch_size"] = int(batch_size)
            st.session_state["max_workers"] = int(max_workers)

        return {
            "model": model,
            "temperature": float(temperature),
            "truncate_chars": int(truncate_chars),
            "chosen": chosen,
            "batch_size": int(batch_size),
            "max_workers": int(max_workers),
        }

    def brand_profile_editor(self, chosen: str) -> tuple[Dict[str, Any], str]:
        # берём профиль из файла или ручной
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

        with st.expander("Бренд-паттерны для предобработки (regex)", expanded=False):
            extra_brand_patterns = st.text_area(
                "Каждая строка — отдельный regex. Добавится к brand_name и aliases.",
                value=st.session_state.get("extra_brand_patterns", ""),
                height=120,
                key="extra_brand_patterns",
                placeholder=r"например:\n(?<!\w)familia(?!\w)\n(?<!\w)фамилия(?!\w)\n",
            )

        with st.expander("Правила «точно drop» (бренд-специфичные регэкспы)"):
            sure_drop_text = st.text_area(
                "По одному паттерну в строке",
                value="\n".join(profile.get("sure_drop_patterns") or []),
                height=140,
                placeholder=r'(?i)\bsagrada\s+familia\b',
            )

        with st.expander("Маркеры PR/официальных ответов (регэкспы)"):
            pr_text = st.text_area(
                "По одному паттерну в строке",
                value="\n".join(profile.get("pr_reply_markers") or []),
                height=120,
                placeholder=r"(?i)^здравствуйте",
            )

        # обновим профиль
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

        st.markdown("</div>", unsafe_allow_html=True)
        return profile, extra_brand_patterns

    def system_prompt_section(self, profile: Dict[str, Any]) -> str:
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
        return final_system

    def ensure_ready(self):
        if _IMPORT_ERR is not None:
            st.error("Не найдены файлы llm_model.py и/или filter_service.py.")
            st.code(str(_IMPORT_ERR), language="text")
            st.stop()

        if not self.api_key_present:
            st.warning(
                "Не найден OPENAI_API_KEY. Добавь ключ в env или Streamlit secrets.")
            # не стопаем — пусть UI открывается, но запуск заблокируем по кнопке

    def render_single(
        self,
        *,
        profile: Dict[str, Any],
        final_system: str,
        extra_brand_patterns: str,
        model: str,
        temperature: float,
        truncate_chars: int,
    ):
        st.subheader("Комментарий (один)")

        single_text = st.text_area(
            "Вставь текст",
            height=180,
            placeholder="Один комментарий сюда…",
            key="single_text",
        )
        run_one = st.button("🚀 Запустить", type="primary",
                            use_container_width=True, key="run_one_btn")
        st.markdown("</div>", unsafe_allow_html=True)

        if not run_one:
            return

        if not self.api_key_present:
            st.error("Нет OPENAI_API_KEY — добавь ключ в Secrets.")
            st.stop()

        # preprocessor под текущий профиль/паттерны
        preproc_factory = PreprocessorFactory(max_words=250)
        pre = preproc_factory.make(profile, extra_brand_patterns)

        def preprocess_fn(text_rule: str) -> str:
            return preproc_factory.preprocess_for_llm(text_rule, pre)

        client = get_client()
        llm = OpenAIRelevanceBatchModel(client=client, default_model=model)
        service = RelevanceFilterService(llm=llm)

        with st.spinner("Фильтрую…"):
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
            total_dt = time.perf_counter() - t0

        action = res.get("action", "keep")
        is_drop = action == "drop"
        source = res.get("source", "llm")

        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.header("Результат")

        if is_drop:
            st.markdown('<span class="badge-drop">DROP</span>',
                        unsafe_allow_html=True)
        else:
            st.markdown('<span class="badge-keep">KEEP</span>',
                        unsafe_allow_html=True)

        if source == "rule":
            st.markdown(' <span class="badge-rule">RULE</span>',
                        unsafe_allow_html=True)
            st.caption(
                f"Pre-LLM правило: {res.get('rule', {}).get('rule_code', 'rule')}")
        else:
            st.caption(
                f"Latency (batch): {res.get('latency_s', 0.0):.3f}s • total: {total_dt:.3f}s")

        st.subheader("JSON")
        st.json({"results": [{"global_idx": 0, "action": action}]})
        st.markdown("</div>", unsafe_allow_html=True)

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
    ):
        st.subheader("Загрузка файла")
        uploaded = st.file_uploader(
            "CSV или Excel. Обязательный столбец: Текст", type=["csv", "xlsx", "xls"])
        run_file = st.button("🚀 Обработать файл",
                             type="primary", use_container_width=True)

        if not run_file:
            return

        if not self.api_key_present:
            st.error("Нет OPENAI_API_KEY — добавь ключ в Secrets.")
            st.stop()

        if uploaded is None:
            st.error("Загрузи файл CSV/XLSX.")
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

        texts = df_in["Текст"].astype(str).tolist()

        preproc_factory = PreprocessorFactory(max_words=250)
        pre = preproc_factory.make(profile, extra_brand_patterns)

        def preprocess_fn(text_rule: str) -> str:
            return preproc_factory.preprocess_for_llm(text_rule, pre)

        client = get_client()
        llm = OpenAIRelevanceBatchModel(client=client, default_model=model)
        service = RelevanceFilterService(llm=llm)

        with st.spinner("Фильтрую (RULE + LLM батчами)…"):
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

        is_drop = ["Yes" if a == "drop" else "No" for a in actions]
        df_out = pd.DataFrame({"Текст": texts, "is_drop": is_drop})

        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Готово")

        st.caption(
            f"Строк: {stats.get('n', len(texts))} • RULE drops: {stats.get('rule_drops', 0)} • "
            f"LLM calls: {stats.get('llm_calls', 0)} • Total: {stats.get('total_s', 0.0):.2f}s • "
            f"comments/s: {stats.get('comments_per_s', 0.0) or 0.0:.2f}"
        )

        st.dataframe(df_out.head(20), use_container_width=True)

        name = (uploaded.name or "result").lower()
        out_fmt = "xlsx" if name.endswith((".xlsx", ".xls")) else "csv"
        out_bytes, mime = df_to_download_bytes(df_out, out_fmt=out_fmt)
        out_name = f"filtered_{Path(uploaded.name).stem}.{out_fmt}"

        st.download_button(
            "⬇️ Скачать результат",
            data=out_bytes,
            file_name=out_name,
            mime=mime,
            use_container_width=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

    def run(self):
        st.title("Relevance Filter")
        st.caption(
            "Карточка бренда + строгие правила «точно drop» перед LLM + предобработка + батчи для файлов.")

        self.ensure_ready()

        settings = self.sidebar_settings()
        profile, extra_brand_patterns = self.brand_profile_editor(
            settings["chosen"])
        final_system = self.system_prompt_section(profile)

        mode = st.radio("Режим", ["Один комментарий",
                        "Файл (CSV/XLSX)"], horizontal=True)

        if mode == "Файл (CSV/XLSX)":
            self.render_file(
                profile=profile,
                final_system=final_system,
                extra_brand_patterns=extra_brand_patterns,
                model=settings["model"],
                temperature=settings["temperature"],
                truncate_chars=settings["truncate_chars"],
                batch_size=settings["batch_size"],
                max_workers=settings["max_workers"],
            )
        else:
            self.render_single(
                profile=profile,
                final_system=final_system,
                extra_brand_patterns=extra_brand_patterns,
                model=settings["model"],
                temperature=settings["temperature"],
                truncate_chars=settings["truncate_chars"],
            )


StreamlitRelevanceApp().run()
