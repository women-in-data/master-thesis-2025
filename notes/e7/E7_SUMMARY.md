# E7 Summary

## Setup
- Model: intfloat/multilingual-e5-small (384-dim)
- PCA explained variance @ 25: 0.4351

## Test metrics (canon split, n=34416)
- ROC-AUC: 0.9285
- PR-AUC:  0.6660
- R@P90:   0.3299
- best_iteration: 499 / 500

## Reference (results_table_v2.csv)
- E0_canon: 0.1721
- E5:       0.2956
- E6_spw15: 0.2078

## Independent CI for E7
- PR-AUC: [0.6454, 0.6868]
- R@P90:  [0.2557, 0.3572]

## Paired comparisons (recall_p90)
- E7 vs e0_canon: delta=+0.1332  CI95=[+0.0363, +0.2389]  significant=True
- E7 vs e5: delta=+0.0308  CI95=[-0.0136, +0.0746]  significant=False
- E7 vs e6_spw15: delta=+0.1075  CI95=[+0.0306, +0.1741]  significant=True
