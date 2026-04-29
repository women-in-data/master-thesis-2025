# Counterfeit Detection Service

Прототип сервиса мультимодальной детекции контрафактных товаров на маркетплейсе Ozon.

## Архитектура

Три модальности → Feature Fusion CatBoost:

```
Фото товара  →  CLIP ViT-B/32 (512-dim)  →  img_scaler
Текст        →  Doc2Vec PV-DM (200-dim)
Табличные    →  38 признаков продавца/товара
                        ↓
              CatBoost Fusion Model
                        ↓
         { is_counterfeit, probability, signals }
```

**Метрики модели:** ROC-AUC 0.9228 | PR-AUC 0.6665 | Recall@P≥0.9 0.0206

## Структура

```
counterfeit_service/
├── app/
│   ├── main.py        # FastAPI роуты
│   ├── predictor.py   # Inference pipeline
│   └── schemas.py     # Pydantic схемы
├── static/
│   └── index.html     # Веб-интерфейс
├── artifacts/         # Файлы моделей (скачать отдельно, см. ниже)
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

## Артефакты моделей

Файлы моделей не хранятся в репозитории из-за большого размера.  
Скачать с Яндекс Диска можно по ссылке: https://disk.yandex.ru/d/aw6epg3MNkQ9vw

| Файл | Описание | Ссылка |
|------|----------|--------|
| `catboost_model.cbm` | Feature Fusion CatBoost (750 features) | [скачать](#) |
| `d2v_model.pkl` | Doc2Vec PV-DM 200-dim (~555 MB) | [скачать](#) |
| `img_scaler.pkl` | StandardScaler для CLIP эмбеддингов | [скачать](#) |
| `feature_cols.pkl` | Список 38 табличных признаков | [скачать](#) |
| `cat_cols.pkl` | Категориальные признаки CatBoost | [скачать](#) |

## Запуск

### Локально

```bash
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

### Docker

```bash
docker compose up --build
```

Открыть: http://localhost:8000  
Swagger UI: http://localhost:8000/docs

## API

`POST /predict` — multipart/form-data:

| Поле | Тип | Описание |
|------|-----|----------|
| `image` | file | Фото товара |
| `name` | string | Название товара |
| `description` | string | Описание |
| `brand` | string | Бренд |
| `category` | string | CommercialTypeName4 |
| `price` | float | Цена |
| `item_time_alive` | float | Дней на площадке |
| `item_count_sales30` | float | Продажи за 30 дней |
| `item_count_returns30` | float | Возвраты за 30 дней |
| `seller_time_alive` | float | Возраст продавца (дни) |

Ответ:

```json
{
  "is_counterfeit": true,
  "probability": 0.87,
  "signals": {
    "multimodal_score": 0.87,
    "image_signal": 0.91,
    "text_signal": 0.43
  }
}
```
