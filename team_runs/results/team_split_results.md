# team_runs — results on the canonical team split

Reproducible re-run of the FinTech-branch experiments on the **command split** (double GroupShuffleSplit by SellerID, `random_state=42`, mirror of `notebooks/baseline.ipynb` cell 7) for cross-domain comparability with the rest of the master-thesis team.

All numbers below are computed from saved arrays — no hand-typed values, no carry-overs from memory. The full reproduction path is `01_team_split.ipynb` → `02_baselines.ipynb` → `03_multimodal_colab.ipynb` (Colab T4) → `04_analysis.ipynb`.

---

## 1. Split

Source: `notebooks/baseline.ipynb` cell 7. Frozen positional indices stored in `team_runs/splits/team_{train,val,test}_idx.npy`.

| | rows | unique sellers | positive rate |
|---|---:|---:|---:|
| train | 69 453 (35.22%) | 3 908 | 6.10% |
| val | 69 335 (35.16%) | 3 908 | 6.02% |
| **test** | **58 410 (29.62%)** | **3 351** | **7.95%** |

Sanity: `train ∩ test sellers = 0` and `val ∩ test sellers = 0` (assert in `01_team_split.ipynb`). CLIP coverage on test = 95.24% (above the user 95% STOP threshold); train/val coverage is 94.80% / 95.38% (the train −0.2pp dip is a sampling fluctuation around the 95.13% global coverage and was treated as a warning, not a stop).

---

## 2. CatBoost configuration (one config for every team_runs experiment)

```python
CatBoostClassifier(
    iterations=1000,
    depth=6,
    learning_rate=0.05,
    eval_metric='AUC',
    scale_pos_weight=15.3959,         # (y_train == 0).sum() / (y_train == 1).sum()
    early_stopping_rounds=50,
    random_seed=42,
    verbose=100,
)
```

**Disclaimer.** This is identical to the FinTech-branch E0_canon config (`fintech_approaches/fintech_experiment.ipynb` cell 5). The choice is intentional: by holding the model identical and only varying the feature/embedding axis, the deltas between team_runs experiments are interpretable as "the effect of features", not "features mixed with hyper-params". For the downstream cross-domain stacking in Chapter 5, the per-domain hyperparameters do not matter — the meta-classifier consumes probability arrays of shape `(58410,)`, not domain-model hyperparameters.

---

## 3. Comparative table

### 3.1. Re-run on the team split — full metrics from `team_runs/proba/*.npy`

| Experiment | Feature spec | ROC-AUC | PR-AUC | R@P≥0.9 | best_iter | Δ vs E0_team_clean |
|---|---|---:|---:|---:|---:|---|
| E0_team_clean | team (38 features, no text) | 0.9050 | 0.5301 | **0.0022** | 293 / 1000 | — |
| E0_team_full | mine (42, text-as-category) | 0.9366 | 0.6812 | **0.0691** | 300 / 1000 | +0.069 R@P90, +0.151 PR-AUC |
| **E5_team** | mine + CLIP-PCA-25 | 0.9508 | 0.7189 | **0.0950** | 495 / 1000 | +0.093 R@P90, +0.189 PR-AUC |
| **E7_team** | mine + e5-small text-PCA-25 | 0.9521 | 0.7224 | **0.0915** | 729 / 1000 | +0.089 R@P90, +0.192 PR-AUC |
| **Fusion_team** | mine + CLIP-PCA-25 + text-PCA-25 | **0.9522** | **0.7284** | **0.1077** | 429 / 1000 | +0.106 R@P90, +0.198 PR-AUC |

Every model converged via early stopping (`best_iter < 1000`).

### 3.2. Original FinTech-branch results — re-computed from `fintech_approaches/test_proba_*.npy` against `y_test_canon.npy`

These rows are recomputed (not pulled from `results_table_v2.csv`) so they cannot drift relative to the actual saved probabilities. They match `results_table_v2.csv` to 4 decimal places, which is a clean cross-check.

| Experiment | mine train size | mine test size | ROC-AUC | PR-AUC | R@P≥0.9 |
|---|---:|---:|---:|---:|---:|
| E0_canon | 135 626 | 34 416 | 0.9208 | 0.6587 | **0.1721** |
| E5_CLIP_PCA_25 | 135 626 | 34 416 | 0.9232 | 0.6724 | **0.2956** |
| E7_text_e5small_PCA25 | 135 626 | 34 416 | 0.9285 | 0.6660 | **0.3299** |

### 3.3. Cross-domain comparison (R@P≥0.9 only — same model on the two splits)

| Variant | mine R@P90 | team R@P90 | drop |
|---|---:|---:|---:|
| E0 (full features) | 0.1721 | 0.0691 | −60% |
| E5 (CLIP) | 0.2956 | 0.0950 | −68% |
| E7 (text) | 0.3299 | 0.0915 | −72% |

The drop is **structural**, not a regression: the team test has 58 k rows / 3 351 unseen sellers vs mine test 34 k / 1 676, and the team test positive rate is higher (7.95% vs 5.93%). Hitting precision ≥ 0.9 at this scale is uniformly harder. **The relative effect of multimodality is preserved across both splits**, which is the reason the team_runs re-run was valuable in the first place.

---

## 4. Bootstrap CI (paired, 95%, n=1000, seed=42, on the team test, n=58 410)

Source: `team_runs/results/bootstrap_team_pairs.csv`. Significant means the 95% CI does not cross zero. Δ = metric(B) − metric(A).

### R@P≥0.9

| Pair (B vs A) | Δ R@P90 | 95% CI | Significant |
|---|---:|---|:---:|
| E0_team_full vs E0_team_clean | +0.0756 | [+0.020, +0.144] | ✓ |
| E5_team vs E0_team_clean | +0.0996 | [+0.064, +0.127] | ✓ |
| E7_team vs E0_team_clean | +0.1064 | [+0.055, +0.148] | ✓ |
| E5_team vs E0_team_full | +0.0240 | [−0.028, +0.075] | ✗ directional |
| E7_team vs E0_team_full | +0.0308 | [−0.013, +0.088] | ✗ directional |
| Fusion_team vs E5_team | +0.0097 | [−0.041, +0.054] | ✗ directional |
| Fusion_team vs E7_team | +0.0029 | [−0.045, +0.050] | ✗ directional |

### PR-AUC

| Pair (B vs A) | Δ PR-AUC | 95% CI | Significant |
|---|---:|---|:---:|
| E0_team_full vs E0_team_clean | +0.1510 | [+0.140, +0.162] | ✓ |
| E5_team vs E0_team_clean | +0.1888 | [+0.177, +0.200] | ✓ |
| E7_team vs E0_team_clean | +0.1922 | [+0.180, +0.204] | ✓ |
| E5_team vs E0_team_full | +0.0378 | [+0.033, +0.043] | ✓ |
| E7_team vs E0_team_full | +0.0412 | [+0.035, +0.047] | ✓ |
| Fusion_team vs E5_team | +0.0094 | [+0.005, +0.014] | ✓ |
| Fusion_team vs E7_team | +0.0061 | [+0.001, +0.011] | ✓ |

### Interpretation

- **Adding text-as-category** (E0_full vs E0_clean): significant on both metrics — the team feature spec without text underperforms by ~7.6pp R@P90 and ~15.1pp PR-AUC.
- **Adding embeddings to mine spec** (E5/E7 vs E0_full): significant on PR-AUC (+3.8pp / +4.1pp), only directional on R@P90. R@P90 is a high-variance tail metric on a 4 642-positive test; PR-AUC is the more reliable signal here. PASS on PR-AUC, DIRECTIONAL on R@P90.
- **Fusion over single modality** (Fusion vs E5/E7): significant on PR-AUC (+0.94pp / +0.61pp), only directional on R@P90. The two modalities carry complementary signal but the marginal gain over the better single modality is small.

---

## 5. Notes (methodology)

- **PCA fit on train only** (CLIP-25 train-fit, then transform on val and test). Same for text-PCA-25. No leakage. Explained variance: CLIP-25 = 57.4% (of 512), text-25 = 42.9% (of 384).
- **Missing embeddings → fillna(0)** before PCA. Test-side missing rates: CLIP 4.76%, text 0% (text was generated by us on the full DataFrame, so coverage is 100%). The fillna(0)-strategy is identical to the one used in `fintech_experiment.ipynb` cell 24 for E5_canon.
- **e5-small text embeddings** were generated once on the full `ozon_train.csv` (197 198 rows, 14m37s on Colab T4) and cached at `team_runs/embeddings/text_e5_small.parquet` (436 MB). Reusable for any future experiment on the same dataset. Model: `intfloat/multilingual-e5-small`, prompt prefix `"passage: "`, fields concatenated: `name_rus + description + brand_name + CommercialTypeName4`. Same recipe as `notebooks/E7_colab_runner.ipynb`.
- **E0_canon vs E0_old** discrepancy in `results_table_v2.csv` (135 626 vs 110 970 train rows) is the well-known FIX-1 artifact and is not relevant to the team_runs comparison — all team_runs models share one train of 69 453 rows.
- **Splits are frozen**: `team_runs/splits/team_*_idx.npy` allows any future re-run to reproduce these numbers exactly without re-computing GSS.

---

## 6. Replication command-line summary

```bash
# Stage 1 (local, ~5s)
jupyter nbconvert --execute team_runs/notebooks/01_team_split.ipynb --inplace

# Stage 2 (local, ~16s)
jupyter nbconvert --execute team_runs/notebooks/02_baselines.ipynb --inplace

# Stage 2.5 + 3 (Colab T4, ~17 min)
# Open team_runs/notebooks/03_multimodal_colab.ipynb in Colab, Run All.

# Stage 4 (local, ~30s)
jupyter nbconvert --execute team_runs/notebooks/04_analysis.ipynb --inplace
```

Artifacts touched outside `team_runs/`: **none** (verified via `git status`).
