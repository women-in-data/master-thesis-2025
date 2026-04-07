# experiment_execution_plan

## Experiments

1. **E0 — Baseline CatBoost**
   - Split: seller-based
   - Threshold: `0.5`

2. **E1 — Baseline CatBoost + threshold optimization**
   - Use validation set to optimize threshold

3. **E2 — Undersampling + calibration + threshold optimization**

4. **E3 — Optional: seller-level deviation features**

## Rule

Не переходить к следующему эксперименту, пока не зафиксированы метрики текущего эксперимента в `results_table` и `experiment_log`.
