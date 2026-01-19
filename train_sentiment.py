# train_sentiment.py
from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Optional, Dict, Any, List

import numpy as np
import pandas as pd
import yaml
from openai import OpenAI

import processing as proc
from embedders import OpenAIEmbedder, SentenceTransformerEmbedder
from sentiment_model import SentimentModel, SentimentModelConfig


def get_openai_key() -> str:
    # 1) ENV
    k = os.getenv("OPENAI_API_KEY")
    if k:
        return k

    # 2) .streamlit/secrets.toml (локально)
    p = Path(".streamlit/secrets.toml")
    if p.exists():
        try:
            import tomllib  # py3.11+
        except Exception:
            tomllib = None

        if tomllib is not None:
            with open(p, "rb") as f:
                data = tomllib.load(f)
            k2 = data.get("OPENAI_API_KEY")
            if k2:
                return str(k2).strip()

    raise SystemExit(
        "OPENAI_API_KEY not found in env or .streamlit/secrets.toml")


# -------- label parsing --------
def parse_label(x) -> Optional[int]:
    """
    Accepts:
      - int 0/1/2
      - str: negative/neutral/positive
      - ru: негатив/нейтрал/нейтрально/позитив
    Returns: 0/1/2 or None
    """
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return None

    if isinstance(x, (int, np.integer)):
        v = int(x)
        return v if v in (0, 1, 2) else None

    s = str(x).strip().lower()

    if s in ("0", "neg", "negative", "негатив", "нег", "плохо", "bad", "-1"):
        return 0
    if s in ("1", "neu", "neutral", "нейтрал", "нейтр", "нейтрально", "ok", "okay", "0.0"):
        return 1
    if s in ("2", "pos", "positive", "позитив", "поз", "хорошо", "good", "+1"):
        return 2

    return None


# -------- file reading --------
def read_table(path: str) -> pd.DataFrame:
    p = path.lower()
    if p.endswith((".xlsx", ".xls")):
        return pd.read_excel(path, engine="openpyxl")
    return pd.read_csv(path, sep=None, engine="python", encoding="utf-8-sig")


# -------- preprocessing --------
def build_sentiment_preprocessor(max_words: int) -> proc.CommentPreprocessor:
    # Для тональности бренд не обязателен: ставим "заглушку", чтобы BRAND_RE компилился.
    brand_patterns = [r"(brand|brands)"]
    return proc.CommentPreprocessor(
        BRAND_PATTERNS=brand_patterns,
        NOISE_PHRASES=proc.NOISE_PHRASES,
        RU_STOP=proc.RU_STOP,
        TOPIC_KEYWORDS=proc.TOPIC_KEYWORDS,
        max_len=max_words,   # max_len в processing.py = число слов
    )


def preprocess_or_fallback(text: str, pre: proc.CommentPreprocessor, max_words: int) -> Optional[str]:
    t = "" if text is None else str(text).strip()
    if not t:
        return None

    out = pre.preprocess(t, max_len=max_words)
    if out:
        return out

    # мягкий fallback: хотя бы базовая чистка
    base = pre.basic_clean_v2(t).strip().lower()
    return base if base else None


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float((y_true == y_pred).mean()) if len(y_true) else 0.0


def main():
    ap = argparse.ArgumentParser()

    # data
    ap.add_argument("--data", required=True,
                    help="Path to CSV/XLSX with texts + labels")
    ap.add_argument("--text-col", default="Текст", help="Column with raw text")
    ap.add_argument("--label-col", default="label",
                    help="Column with labels (0/1/2 or strings)")

    # output
    ap.add_argument("--out-dir", default="sentiment_assets",
                    help="Where to store artifacts")
    ap.add_argument("--name", default="sentiment_default",
                    help="Artifact base name (without extension)")

    # prototype params
    ap.add_argument("--k-per-class", type=int, default=3)
    ap.add_argument("--normalize", action="store_true", default=True)

    # preprocessing
    ap.add_argument("--max-words", type=int, default=250,
                    help="Max words after preprocessing")

    # thresholds
    ap.add_argument("--th-neg", type=float, default=0.7)
    ap.add_argument("--th-neu", type=float, default=0.6)
    ap.add_argument("--th-pos", type=float, default=0.6)

    # embeddings provider
    ap.add_argument("--embed-provider",
                    choices=["openai", "st"], default="openai")
    ap.add_argument("--openai-embed-model", default="text-embedding-3-small")
    ap.add_argument("--openai-dimensions", type=int, default=None)
    ap.add_argument("--st-model", default="Qwen/Qwen3-Embedding-0.6B")
    ap.add_argument("--st-device", default=None, help="cuda/cpu (optional)")
    ap.add_argument("--embed-batch", type=int, default=128)

    args = ap.parse_args()

    # 1) read
    df = read_table(args.data)
    if args.text_col not in df.columns:
        raise SystemExit(
            f"Text column '{args.text_col}' not found. Available: {list(df.columns)}")
    if args.label_col not in df.columns:
        raise SystemExit(
            f"Label column '{args.label_col}' not found. Available: {list(df.columns)}")

    # 2) parse labels
    y_parsed = df[args.label_col].apply(parse_label)
    mask = y_parsed.notna()
    df = df.loc[mask].copy()
    y = y_parsed.loc[mask].astype(int).to_numpy(dtype=np.int64)

    # 3) preprocess texts
    pre = build_sentiment_preprocessor(max_words=int(args.max_words))
    texts_raw = df[args.text_col].astype(str).tolist()

    texts: List[str] = []
    y2: List[int] = []
    for t, lab in zip(texts_raw, y):
        ptxt = preprocess_or_fallback(t, pre, max_words=int(args.max_words))
        if ptxt is None:
            continue
        texts.append(ptxt)
        y2.append(int(lab))

    if not texts:
        raise SystemExit("No texts left after preprocessing.")

    y2 = np.asarray(y2, dtype=np.int64)

    print(f"Loaded rows (after label filter): {len(df)}")
    print(f"After preprocessing: {len(texts)}")
    print("Class counts:", {i: int((y2 == i).sum()) for i in (0, 1, 2)})

    # 4) choose embedder
    embed_meta: Dict[str, Any]
    if args.embed_provider == "openai":
        api_key = get_openai_key()
        client = OpenAI(api_key=api_key)

        embedder = OpenAIEmbedder(
            client=client,
            model=args.openai_embed_model,
            dimensions=args.openai_dimensions,
        )
        embed_fn = embedder.embed_texts
        embed_meta = {"provider": "openai", "model": args.openai_embed_model,
                      "dimensions": args.openai_dimensions}
        openai_key_for_model = api_key  # не обязателен, но ок
    else:
        embedder = SentenceTransformerEmbedder(
            model_name=args.st_model, device=args.st_device)
        embed_fn = embedder.embed_texts
        embed_meta = {"provider": "sentence-transformers",
                      "model": args.st_model, "device": embedder.device}
        openai_key_for_model = None  # не нужен для обучения

    # 5) embeddings
    X = embed_fn(texts, batch_size=int(args.embed_batch))

    # 6) fit prototypes
    cfg = SentimentModelConfig(
        k_per_class=int(args.k_per_class),
        normalize=bool(args.normalize),
        thresholds={0: float(args.th_neg), 1: float(
            args.th_neu), 2: float(args.th_pos)},
        enable_llm_fallback=False,  # при обучении не нужен
    )
    model = SentimentModel(embed_fn=embed_fn, config=cfg,
                           openai_api_key=openai_key_for_model)
    model.fit(X, y2)

    # 7) quick sanity metrics (train)
    pred3, sim_pred, _ = model.predict_from_embeddings(X)
    _, pass_mask = model.apply_thresholds(pred3, sim_pred)
    print(f"Train acc (argmax): {accuracy(y2, pred3):.4f}")
    print(f"Coverage by thresholds: {float(pass_mask.mean()):.4f}")

    # 8) save artifacts
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    npz_path = out_dir / f"{args.name}.npz"
    yml_path = out_dir / f"{args.name}.yaml"

    model.save_artifacts(str(npz_path))

    meta = {
        "k_per_class": int(args.k_per_class),
        "normalize": bool(args.normalize),
        "thresholds": {"negative": float(args.th_neg), "neutral": float(args.th_neu), "positive": float(args.th_pos)},
        "embedding": embed_meta,
        "max_words": int(args.max_words),
        "text_col": args.text_col,
        "label_col": args.label_col,
        "rows_used": int(len(texts)),
    }

    with open(yml_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(meta, f, allow_unicode=True, sort_keys=False)

    print("Saved:")
    print(" -", npz_path)
    print(" -", yml_path)


if __name__ == "__main__":
    main()
