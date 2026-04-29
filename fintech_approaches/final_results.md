## Final Results

| experiment | threshold | roc_auc | pr_auc | precision | recall | recall_high_precision |
| --- | --- | --- | --- | --- | --- | --- |
| E0_baseline_default_threshold | 0.500000 | 0.911635 | 0.642136 | 0.313542 | 0.820588 | 0.219608 |
| E1_threshold_optimization | 0.975285 | 0.911635 | 0.642136 | 0.930851 | 0.085784 | 0.085784 |
| E2_undersampling_calibration | 0.973323 | 0.916622 | 0.648021 | 0.900000 | 0.119118 | 0.119118 |
| E3_deviation_features | 0.976266 | 0.918986 | 0.652982 | 0.936709 | 0.072549 | 0.072549 |

## Best Experiment

- E0_baseline_default_threshold
- recall_high_precision = 0.219608

## Summary

## SUMMARY (для диплома)

Исследовательский вопрос состоял в том, можно ли повысить качество выявления контрафакта в зоне высокой точности на seller-based split, не нарушая ограничение Precision >= 0.9. Для этого сравнивались четыре постановки на одном и том же разбиении и с одной и той же базовой моделью.

В E0 был использован baseline CatBoost с порогом 0.5. В E1 применялась только threshold optimization на validation set. В E2 использовались undersampling на train и Dal Pozzolo-style calibration, после чего threshold снова подбирался на validation. В E3 к базовому пайплайну были добавлены deviation features, построенные только по train seller statistics.

Главные результаты таковы: recall_high_precision составил 0.219608 для E0, 0.085784 для E1, 0.119118 для E2 и 0.072549 для E3. Лучшим экспериментом по основному критерию оказался E0_baseline_default_threshold со значением recall_high_precision = 0.219608.

Интерпретация результатов следующая: threshold optimization в E1 меняет только decision rule, не изменяя сами вероятности модели. Calibration в E2 корректирует вероятности после обучения на undersampled train, чтобы они лучше соответствовали исходному prior положительного класса. Deviation features в E3 представляют собой feature engineering, которое добавляет относительные отклонения объекта от seller-level train medians.

Финальный вывод: улучшить Recall@Precision>=0.9 относительно baseline не удалось, поскольку лучший результат показал эксперимент E0_baseline_default_threshold. Следовательно, максимальное значение high-precision recall в текущем наборе экспериментов обеспечено baseline-моделью без дополнительных модификаций.
