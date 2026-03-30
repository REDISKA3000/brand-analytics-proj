https://brand-analytics-proj-d9enuniaul4vemjntbhqnv.streamlit.app
# Brand Opinion Analytics 🧭

A project for end-to-end brand opinion analytics from user comments — turning raw UGC into clean, actionable signals for product and marketing.

Architecture overview: see `ARCHITECTURE.md`.

This repository already includes the first module: a **relevance filter** (what’s actually about the brand vs. noise).  
Next in the roadmap: **sentiment analysis** and **semantic tagging**.

---

## What problem this solves

Brand-related comments are both gold and garbage:
- some messages are truly about purchases, service, assortment, pricing, or the app
- others are job posts, “next to the store…” event ads, homonyms, canned PR replies, or random noise

This project builds a simple pipeline:

1) **Relevance filtering** (keep only comments that are actually about the brand)  
2) **Sentiment analysis** (positive/negative/neutral, etc.)  
3) **Semantic tags** (topics like assortment, service, delivery, pricing, app, quality, etc.)

---

## What’s implemented now (MVP)

### ✅ Module 1: Relevance Filter (done)
A Streamlit app that:
- accepts **a single comment** or **a CSV/XLSX file**
- expects a column named **`Текст`** (Russian: “Text”) containing comments
- returns `KEEP / DROP` (for files — `is_drop: Yes/No`)
- saves tokens by applying **hard “sure drop” rules before the LLM**
- improves contextual understanding via a **brand card** + **text preprocessing**

---

## What’s coming next

### 🔜 Module 2: Sentiment (in progress)
Sentiment classification on top of relevant comments:
- sentiment labels
- aggregates over time/sources
- quick exports: “what hurts” vs. “what works”

### 🔜 Module 3: Semantic Tags (in progress)
Topic/aspect tagging for comments:
- one or multiple tags per comment
- taxonomy per brand/industry
- easy taxonomy extension without rewriting core logic

---

## Project principles

- **Stability**: deterministic “sure drop” rules before the LLM
- **Cost efficiency**: don’t spend tokens on cases that are obvious
- **Flexibility**: switch brands via brand card + YAML/manual mode
- **Clarity**: simple output fields ready for analytics/BI
- **Extensibility**: each step is a separate module/class

---

## Brand rules are per‑brand, not hardcoded

All “sure drop” logic is driven by the brand profile in `brands.yaml` (or manual input in the UI).  
The code is brand‑agnostic and supports user-defined `drop_categories`, where each category has:

- `name` — пользовательское название категории
- `description` — описание того, что нужно выбрасывать
- `patterns` — сгенерированные или заданные regex

For a new brand, copy `brands_template.yaml` and fill in your patterns.

---

## Quick start (local)

### 1) Install
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
