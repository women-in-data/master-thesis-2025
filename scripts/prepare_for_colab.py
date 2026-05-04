"""
Prepare data for Colab E7 experiment.

Запусти ОДИН РАЗ локально:
    python prepare_for_colab.py

Что делает:
1. Грузит твой ozon_train.csv
2. Делает canon-split ровно как Cell 1 в fintech_experiment.ipynb
3. Сохраняет train/val/test как отдельные CSV
4. Сохраняет список feature_cols_e5 (42 колонки intersection)
5. Копирует уже посчитанные test_proba_*.npy для bootstrap-сравнений в колабе
6. Упаковывает всё в e7_colab_input.zip

В итоге:
    /Users/sofya/women-in-data-thesis/e7_colab_input.zip   <-- этот файл загружаешь в колаб
"""

import os
import json
import shutil
import zipfile
from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# ============================================================
# CONFIG — поменяй пути если они другие
# ============================================================
RAW_CSV = "/Users/sofya/Desktop/diplomahse/ozon_train.csv"
PROBA_DIR = "/Users/sofya/women-in-data-thesis/fintech_approaches"
OUTPUT_DIR = "/Users/sofya/women-in-data-thesis/e7_colab_prep"
OUTPUT_ZIP = "/Users/sofya/women-in-data-thesis/e7_colab_input.zip"

PROBAS_TO_INCLUDE = [
    "test_proba_e0_canon.npy",
    "test_proba_e5.npy",
    "test_proba_e6_spw15.npy",
    "y_test_canon.npy",
]

# ============================================================
# STEP 1: Load + canon split
# ============================================================
print("=" * 60)
print("STEP 1: Loading data and reproducing canon split (Cell 1)")
print("=" * 60)

assert os.path.exists(RAW_CSV), f"Не нашёл {RAW_CSV} — поменяй RAW_CSV в начале скрипта"
df = pd.read_csv(RAW_CSV)
print(f"Loaded df: {df.shape}, positive_rate={df['resolution'].mean():.4f}")


def make_seller_split(df, seller_col="SellerID", random_state=42):
    """Точная копия Cell 1 из fintech_experiment.ipynb."""
    sellers = df[seller_col].drop_duplicates()
    train_sellers, temp_sellers = train_test_split(
        sellers, test_size=0.30, random_state=random_state, shuffle=True
    )
    val_sellers, test_sellers = train_test_split(
        temp_sellers, test_size=0.50, random_state=random_state, shuffle=True
    )
    train_df = df[df[seller_col].isin(train_sellers)].copy()
    val_df = df[df[seller_col].isin(val_sellers)].copy()
    test_df = df[df[seller_col].isin(test_sellers)].copy()
    return train_df, val_df, test_df


train_df, val_df, test_df = make_seller_split(df)

print(f"  train: {train_df.shape}  positive_rate={train_df['resolution'].mean():.4f}")
print(f"  val:   {val_df.shape}    positive_rate={val_df['resolution'].mean():.4f}")
print(f"  test:  {test_df.shape}   positive_rate={test_df['resolution'].mean():.4f}")

# Sanity check — должны совпадать с FIX-1 числами
assert len(train_df) == 135626, f"train size mismatch: {len(train_df)} != 135626"
assert len(val_df) == 27156, f"val size mismatch: {len(val_df)} != 27156"
assert len(test_df) == 34416, f"test size mismatch: {len(test_df)} != 34416"
print("  ✅ Sanity check passed: split sizes match FIX-1")


# ============================================================
# STEP 2: Apply add_fintech_features (как Cell 2)
# ВАЖНО: Cell 2 применяет ТОЛЬКО к train_df и test_df (НЕ к val_df!)
# Это и создаёт inconsistency #4 — feature_cols_e5 = intersection
# Воспроизводим точно для honest mirror E5
# ============================================================
print()
print("=" * 60)
print("STEP 2: Replicating Cell 2 feature engineering (preserves intersection logic)")
print("=" * 60)


def add_fintech_features(df):
    df = df.copy()
    df["return_rate_30"] = df["item_count_returns30"] / (df["item_count_sales30"] + 1)
    df["fake_return_rate_30"] = df["item_count_fake_returns30"] / (df["item_count_sales30"] + 1)
    df["fake_return_rate_90"] = df["item_count_fake_returns90"] / (df["item_count_sales90"] + 1)
    df["seller_velocity"] = df["item_count_sales30"] / (df["seller_time_alive"] + 1)
    df["gmv_per_day"] = df["GmvTotal30"] / (df["item_time_alive"] + 1)
    df["both_new"] = ((df["item_time_alive"] < 30) & (df["seller_time_alive"] < 90)).astype(int)
    return df


# Только train_df и test_df, НЕ val_df (точная репликация Cell 2)
train_df = add_fintech_features(train_df)
test_df = add_fintech_features(test_df)
# val_df остаётся БЕЗ fintech-фичей

# seller_stats merge — тоже только train и test (как в Cell 2)
seller_stats = train_df.groupby("SellerID").agg(
    seller_item_count=("ItemID", "count"),
    seller_avg_return_rate=("return_rate_30", "mean"),
    seller_avg_fake_returns=("fake_return_rate_30", "mean"),
).reset_index()

train_df = train_df.merge(seller_stats, on="SellerID", how="left")
test_df = test_df.merge(seller_stats, on="SellerID", how="left")
test_df[["seller_item_count", "seller_avg_return_rate", "seller_avg_fake_returns"]] = test_df[
    ["seller_item_count", "seller_avg_return_rate", "seller_avg_fake_returns"]
].fillna(0)

# val_df НЕ трогаем — эмулируем баг Cell 2

print(f"  train_df after FE: {train_df.shape}")
print(f"  val_df after FE:   {val_df.shape}  (intentionally NOT enriched, preserves intersection logic)")
print(f"  test_df after FE:  {test_df.shape}")


# ============================================================
# STEP 3: Build feature_cols_e5 (intersection like Cell 26)
# ============================================================
print()
print("=" * 60)
print("STEP 3: Building feature_cols_e5 (intersection train ∩ val ∩ test)")
print("=" * 60)

base_cols_train = [c for c in train_df.columns if c not in ["resolution", "ItemID", "SellerID"]]
base_cols_val = [c for c in val_df.columns if c not in ["resolution", "ItemID", "SellerID"]]
base_cols_test = [c for c in test_df.columns if c not in ["resolution", "ItemID", "SellerID"]]

feature_cols_e5 = sorted(set(base_cols_train) & set(base_cols_val) & set(base_cols_test))

print(f"  feature_cols_e5: {len(feature_cols_e5)} columns (expected 42 per Inconsistency #4)")
print(f"  Excluded from intersection (only in train): "
      f"{sorted(set(base_cols_train) - set(feature_cols_e5))}")

assert len(feature_cols_e5) == 42, f"feature_cols_e5 should be 42, got {len(feature_cols_e5)}"
print("  ✅ feature_cols_e5 = 42, matches E5 in original notebook")


# ============================================================
# STEP 4: Save split data + feature_cols + probas
# ============================================================
print()
print("=" * 60)
print("STEP 4: Saving everything to OUTPUT_DIR")
print("=" * 60)

os.makedirs(OUTPUT_DIR, exist_ok=True)

# CSV files (need ItemID, SellerID, resolution + feature_cols + text fields)
text_cols = ["name_rus", "description", "brand_name", "CommercialTypeName4"]
required_keep = ["ItemID", "SellerID", "resolution"] + text_cols + feature_cols_e5
required_keep = list(dict.fromkeys(required_keep))  # dedupe preserving order

# Для val_df некоторые feature_cols_e5 могут отсутствовать (если они из train-only)
# Но мы взяли intersection, значит все feature_cols_e5 есть в val
for col in feature_cols_e5:
    assert col in val_df.columns, f"{col} missing in val_df — intersection broken"

train_keep = [c for c in required_keep if c in train_df.columns]
val_keep = [c for c in required_keep if c in val_df.columns]
test_keep = [c for c in required_keep if c in test_df.columns]

train_df[train_keep].to_csv(f"{OUTPUT_DIR}/train_canon.csv", index=False)
val_df[val_keep].to_csv(f"{OUTPUT_DIR}/val_canon.csv", index=False)
test_df[test_keep].to_csv(f"{OUTPUT_DIR}/test_canon.csv", index=False)

print(f"  train_canon.csv: {len(train_df)} rows × {len(train_keep)} cols")
print(f"  val_canon.csv:   {len(val_df)} rows × {len(val_keep)} cols")
print(f"  test_canon.csv:  {len(test_df)} rows × {len(test_keep)} cols")

# feature_cols_e5 metadata
with open(f"{OUTPUT_DIR}/feature_cols_e5.json", "w") as f:
    json.dump(feature_cols_e5, f, indent=2)
print(f"  feature_cols_e5.json: {len(feature_cols_e5)} cols")

# Reference probas for bootstrap comparisons in Colab
proba_dest = f"{OUTPUT_DIR}/reference_probas"
os.makedirs(proba_dest, exist_ok=True)
for proba_file in PROBAS_TO_INCLUDE:
    src = os.path.join(PROBA_DIR, proba_file)
    if os.path.exists(src):
        shutil.copy(src, proba_dest)
        print(f"  reference_probas/{proba_file} ✅")
    else:
        print(f"  reference_probas/{proba_file} ❌ NOT FOUND — skipping")

# README inside zip
readme_text = f"""# E7 Colab Input

Подготовлено для запуска E7 (sentence-transformers + PCA + CatBoost) в Google Colab.

## Что внутри
- `train_canon.csv` ({len(train_df)} rows) — canon train, после Cell 2 FE
- `val_canon.csv`   ({len(val_df)} rows)  — canon val (БЕЗ fintech FE — точная репликация Cell 2 inconsistency)
- `test_canon.csv`  ({len(test_df)} rows)  — canon test, после Cell 2 FE
- `feature_cols_e5.json` — список 42 фич (intersection), используется E7 как зеркало E5
- `reference_probas/` — probas для bootstrap-сравнений в колабе

## Как использовать
1. Загрузи весь zip в колаб через Files panel
2. Распакуй: `!unzip e7_colab_input.zip`
3. Запусти колаб-промпт (отдельный файл)
"""
with open(f"{OUTPUT_DIR}/README.md", "w") as f:
    f.write(readme_text)


# ============================================================
# STEP 5: Zip everything
# ============================================================
print()
print("=" * 60)
print("STEP 5: Creating final zip")
print("=" * 60)

with zipfile.ZipFile(OUTPUT_ZIP, "w", zipfile.ZIP_DEFLATED) as zf:
    for root, _, files in os.walk(OUTPUT_DIR):
        for file in files:
            full_path = os.path.join(root, file)
            arcname = os.path.relpath(full_path, OUTPUT_DIR)
            zf.write(full_path, arcname)
            print(f"  + {arcname}")

zip_size_mb = os.path.getsize(OUTPUT_ZIP) / (1024 * 1024)
print(f"\n✅ DONE: {OUTPUT_ZIP}  ({zip_size_mb:.1f} MB)")
print()
print("Следующий шаг:")
print(f"  1. Открой colab.research.google.com")
print(f"  2. Новый ноутбук → Runtime → Change runtime type → T4 GPU")
print(f"  3. Загрузи {OUTPUT_ZIP} через левую панель Files")
print(f"  4. Скопируй колаб-промпт (e7_colab_runner.py) в первую ячейку")
print(f"  5. Run (всё ячейка целиком)")
