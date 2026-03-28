import pandas as pd
from app import load_brands, format_system_prompt, PreprocessorFactory, df_to_download_bytes, BASE_SYSTEM_TEMPLATE
from filter_service import RuleEngine

# --- 1. Загрузка брендов и промптов ---
brands = load_brands("brands.yaml")
prompts = {k: format_system_prompt(BASE_SYSTEM_TEMPLATE, v) for k, v in brands.items()}

# --- 2. Загрузка комментариев ---
comments_df = pd.read_excel("comments.xlsx")  # или pd.read_csv(...)

# --- 3. Обработка комментариев ---
pre_factory = PreprocessorFactory()
rule_engine = RuleEngine()
results = []
for idx, row in comments_df.iterrows():
    brand = row.get('brand', '')
    profile = brands.get(brand, {})
    pre = pre_factory.make(profile, "")
    text = row.get('comment', '')
    rule = rule_engine.sure_drop(text, profile)
    results.append({
        "brand": brand,
        "comment": text,
        "is_relevant": "drop" if rule else "keep"
    })

# --- 4. Сохранение результатов в Excel ---
result_df = pd.DataFrame(results)
bytes_data, mime = df_to_download_bytes(result_df, "xlsx")
with open("result.xlsx", "wb") as f:
    f.write(bytes_data)

print("Готово! Результаты сохранены в result.xlsx")
