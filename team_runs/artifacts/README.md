# `team_runs/artifacts/` — inference-артефакты Sonya FT-MFF (Fusion_team)

**Назначение.** Production-ready артефакты модели **FT-MFF** (FinTech Multi-Modal Fusion) — она же `Fusion_team` из моей Главы 4 и `Channel 1` (fusion-канал) общекомандной CDSM v3 из общей Главы 5. Эти файлы нужны для **инференса** на новых товарных карточках (например, в сервисе на Timeweb или в командном пайплайне).

Сгенерировано локально 12 мая 2026 ноутбуком [`../notebooks/03b_fusion_retrain_local.ipynb`](../notebooks/03b_fusion_retrain_local.ipynb) (CPU retrain). Original оригинальная модель училась на Colab T4 в `03_multimodal_colab.ipynb`, но `.cbm` файла там не сохранилось (только probas) — поэтому переобучила локально для передачи в команду.

## Файлы

| Файл | Размер | Что это |
|---|---|---|
| `ftmff_catboost.cbm` | 23.8 МБ | Обученная CatBoost-модель Fusion_team. Загружается через `CatBoostClassifier().load_model('ftmff_catboost.cbm')`. |
| `ftmff_pca_clip.pkl` | 54 КБ | Зафиченный `sklearn.decomposition.PCA(25)` на 512-dim CLIP ViT-B/32 эмбеддингах. Нужен для трансформации новых CLIP-эмбеддингов перед подачей в CatBoost. |
| `ftmff_pca_text.pkl` | 41 КБ | Зафиченный `sklearn.decomposition.PCA(25)` на 384-dim multilingual-e5-small эмбеддингах. Нужен для трансформации новых text-эмбеддингов перед подачей в CatBoost. |
| `test_proba_fusion_team_LOCAL_retrain.npy` | 234 КБ | Probas локально переобученной модели на командном тесте. Для сравнения с canonical `../proba/test_proba_fusion_team.npy`. |
| `ftmff_catboost_summary.json` | < 1 КБ | Метаданные обучения, метрики, корреляция с canonical, список артефактов. |

## Important: расхождение с canonical

Локально переобученная модель **не идентична побитно** оригинальной (Colab T4). Pearson-корреляция probas = **0.989**, max absolute diff = **0.45**. Метрики **в пределах bootstrap CI**:

| Метрика | Canonical (Colab T4) | New (local CPU) |
|---|---|---|
| ROC-AUC | 0.9522 | 0.9572 |
| PR-AUC | 0.7284 | 0.7279 |
| R@P≥0.9 | 0.1077 | 0.1030 |

Причина расхождения — float32 numerical differences CPU vs GPU + разные версии библиотек. **Это нормально для CatBoost cross-platform**, не баг.

**⚠️ В пайплайне расчёта ансамблей (CDSM, B0 blending) использовать только canonical [`../proba/test_proba_fusion_team.npy`](../proba/test_proba_fusion_team.npy), не локальный retrain.** Локальный файл — только для проверки эквивалентности модели.

## Pipeline применения модели к новой товарной карточке

```python
import numpy as np
import joblib
from catboost import CatBoostClassifier
from sentence_transformers import SentenceTransformer

ART = 'team_runs/artifacts'
model = CatBoostClassifier().load_model(f'{ART}/ftmff_catboost.cbm')
pca_clip = joblib.load(f'{ART}/ftmff_pca_clip.pkl')
pca_text = joblib.load(f'{ART}/ftmff_pca_text.pkl')
e5 = SentenceTransformer('intfloat/multilingual-e5-small')

# Для нового товара X с уже посчитанными CLIP-512 и text e5-384 эмбеддингами,
# и табличными признаками tabular_df (42 столбца, как в Главе 4 § 4.2.2):
clip_pca_25 = pca_clip.transform(clip_emb_512.reshape(1, -1))      # 1×25
text_pca_25 = pca_text.transform(text_emb_384.reshape(1, -1))      # 1×25
X = pd.concat([
    tabular_df,                                                     # 42 cols
    pd.DataFrame(clip_pca_25, columns=[f'clip_pca_{i}' for i in range(25)]),
    pd.DataFrame(text_pca_25, columns=[f't_pca_{i}'    for i in range(25)]),
], axis=1)                                                          # 92 cols total

proba = model.predict_proba(X)[:, 1]
```

**Cat features:** список категориальных признаков в `MY_CATS` ноутбука (4 текстовых поля: `brand_name`, `description`, `name_rus`, `CommercialTypeName4`). Они подаются в CatBoost как строковые категории с встроенным Ordered Target Statistics encoding.

## Как воспроизвести

```bash
cd <repo>
python3 -c "
import json, sys
sys.path.insert(0, 'team_runs/notebooks')
exec(open('/Users/sofya/.../run_fusion_retrain.py').read())  # или запустить ноутбук 03b
"
```

Или открыть `team_runs/notebooks/03b_fusion_retrain_local.ipynb` и выполнить все ячейки. Время ~30 сек на маке (CPU).

**Требования:**
- `ozon_train.csv` (190 МБ, Ozon Sustainability data)
- `clip_embeddings.parquet` (591 МБ, готовые CLIP ViT-B/32)
- `text_e5_small.parquet` (кэшируется автоматически при первом запуске, ~30 мин на CPU или 5 мин на GPU)
- `team_train_idx.npy`, `team_val_idx.npy`, `team_test_idx.npy`, `y_test.npy` (команд­ный сплит)

## Контекст и ссылки

- **Моя Глава 4 § 4.4** — Fusion_team как объединение трёх модальностей
- **Общая Глава 5 § 5.4.5.2** — этот же fusion как Channel 1 ансамбля CDSM v3
- **`team_runs/results/stacking_pilot_4domains.md`** — кросс-доменный пилот, где Fusion_team участвует как одиночка и в блендинге B0

## Acknowledgments

Локальный retrain — для передачи в команду (Карина / Диана / Альбина) на 12 мая 2026, по запросу Дианы для нового сервиса деплоя CDSM v3.
