# Domain Summary: Mobile Apps & Advertising (Azimova K.)

## Dataset & Split

- **Ozon eCup 2025**: 197,198 items, 45 features, positive rate 6.6%
- **Seller-based split** (GroupShuffleSplit by SellerID, random_state=42): train=177,380 / test=19,818
- Constraint: SellerID(train) ∩ SellerID(test) = ∅ — models are evaluated on sellers never seen during training
- **Key metric**: Recall@P≥0.9 — share of counterfeit items detected at precision ≥ 0.90

## Baselines (Chapter 3)

| Model | ROC-AUC | PR-AUC | R@P≥0.9 |
|---|---|---|---|
| B1: TF-IDF + LR (text only) | 0.905 | 0.590 | 0.000 |
| B2: CatBoost (38 tab features) | 0.905 | 0.623 | 0.015 |
| B3: 32-dim image embeddings + LR | 0.853 | 0.462 | 0.004 |

No single-modality model achieves practically useful R@P≥0.9.

## Experiment Progression (Chapter 4)

| Model | Feature vector | R@P≥0.9 | PR-AUC |
|---|---|---|---|
| Exp-1 | 38 tab + 512 CLIP + 50 TF-IDF SVD = **600-dim** | 0.097 | 0.687 |
| Exp-Deng | Exp-1 + 3 typosquat + 1 clip_cat_sim = **604-dim** | 0.190 | 0.737 |
| Exp-FADAML | Exp-1 + 3 price + 8 domain text = **611-dim** | 0.222 | 0.739 |
| **Exp-Combined** | Exp-1 + Deng + FADAML = **614-dim** | **0.224** | **0.745** |

Classifier: CatBoost, iterations=1500, depth=7, lr=0.05, early_stopping=100, random_seed=42, scale_pos_weight=N0/N1.

## What This Domain Contributes to the Team Model

### Block 1: Feature Fusion architecture
Early concatenation of all modality vectors into a single matrix, single CatBoost on top.
Feature Fusion > Late Fusion: +3.2 pp PR-AUC, +0.032 R@P≥0.9 at identical feature sets.

### Block 2: CLIP ViT-B/32 (512-dim)
Replacing 32-dim embeddings with CLIP: R@P≥0.9 from 0.013 → 0.097 (×7.5).
StandardScaler fit on train only. Zero imputation for missing embeddings (~0.4% of items).

### Block 3: TF-IDF + TruncatedSVD (50 components)
Fields: name_rus + description + brand_name, ngram_range=(1,2), max_features=50k, min_df=5.
Bigrams are critical: typosquat pattern "samsung" vs "samsang" is invisible at unigram level.

### Block 4: Deng et al. typosquatting features (4 features)
- `brand_exact`: exact match of brand_name in name_rus (binary)
- `brand_fuzzy`: partial_ratio(brand, name) / 100 via rapidfuzz
- `typosquat`: max(0, brand_fuzzy − 0.5 × brand_exact)
- `clip_cat_sim`: cosine similarity of item CLIP embedding to category centroid (fit on train)

Marginal contribution over Exp-1: +0.093 R@P≥0.9 (+96%).

### Block 5: FADAML price & domain text features (11 features)
**Price (3)**: price_ratio = PriceDiscounted / (category_median + 1), price_too_low (<0.3), price_too_high (>0.9)  
**Domain text (8)**: susp_kw (14 authenticity keywords), desc_len, caps_ratio, excl_count, digits_count, brand_exact, brand_fuzzy, typosquat  
Marginal contribution over Exp-1: +0.125 R@P≥0.9 (+129%).

## Negative Results — What NOT to Use

| Approach | Result | Reason |
|---|---|---|
| SAFE (text↔image cosine similarity) | Δ < 0.2σ between classes | Counterfeiters copy both text and image → maximize cross-modal agreement |
| seller_fraud_rate | r=+0.794 on train, undefined on test | Seller-based split: 100% of test sellers are unseen |
| Doc2Vec / BERT | −20.1 pp ROC-AUC vs TF-IDF | Semantic smoothing collapses typosquat signal |
| KL-divergence (CAFE-style) | Degrades on test distribution | Train/test distribution shift in out-of-fold predictions |
| EDA features (fake_return_rate, is_null_*) | ×6.3 on unimodal, ~0 in multimodal | CLIP and TF-IDF implicitly encode the same signal (SHAP ranks 608–623/626) |
| scale_pos_weight tuning | PR-AUC −15.9 pp | Shifts predictions to [0.4, 0.6], destroys ranking at high precision |

## Methodological Constraints for Team Model

1. **Seller-based split is mandatory.** Random stratified split causes leakage of seller-level aggregates. Any feature aggregated at the entity level (seller, user, account) is undefined for test entities in a proper split.
2. **Do not use semantic encoders (Doc2Vec, BERT) for fraud detection.** Anywhere exact tokens matter (brand names, identifiers, property names), semantic smoothing destroys the critical signal.
3. **EDA-derived features need ablation in the full multimodal vector**, not just in the unimodal baseline — their marginal value drops to near zero when CLIP+TF-IDF are present.

## Team Split Metrics (test = 58,410, Sonya's canonical split)

`test_proba_karina_team.npy` (shape: 58410,):

| ROC-AUC | PR-AUC | R@P≥0.9 |
|---|---|---|
| 0.9407 | 0.6562 | 0.0666 |
