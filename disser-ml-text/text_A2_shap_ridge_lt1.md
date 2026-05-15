# Приложение А-2. SHAP-проекция LT1-победителей на Ridge LinearExplainer

Основной текст подраздела 3.7 интерпретирует фактического победителя LT1 — SVR с радиальным базисным ядром (C = 100, ε = 1,0) — через PermutationExplainer. Настоящее приложение приводит **линейную Ridge-проекцию** тех же двух конфигураций (Ridge α = 1000 с LinearExplainer), что даёт согласованный с LT2-таблицами 3.13–3.14 формат и позволяет читателю сопоставить две интерпретации одной и той же предсказательной задачи.

Линейная Ridge-проекция корректно отражает структуру вкладов только в той мере, в какой нелинейная модель SVR может быть аппроксимирована линейной комбинацией признаков. Соответственно, числовые ранги отдельных признаков в Ridge-проекции и SVR-PermutationExplainer не совпадают, хотя качественные выводы (отсутствие доминирующего признака, доля EMG > NIRS > kinematics для LT1 / EMG+NIRS, упорядоченность групп EMG > NIRS > HRV > interaction > kinematics для LT1 / EMG+NIRS+HRV) согласуются.

## А-2.1 LT1 / EMG+NIRS (Ridge-проекция)

**Таблица А-2.1.** Топ-10 признаков по SHAP-важности для Ridge (α=1000) / EMG+NIRS / LT1.

| Признак | Группа | mean \|SHAP\| | Доля, % |
|---------|--------|---------------|---------|
| `z_vl_dist_load_wavelet_entropy` | EMG | 48,33         | 4,9 |
| `trainred_hbdiff_std` | NIRS | 44,46         | 4,5 |
| `trainred_smo2_drop` | NIRS | 44,33         | 4,5 |
| `trainred_hhb_mean` | NIRS | 43,07         | 4,3 |
| `hhb_from_running_min` | NIRS | 39,05         | 3,9 |
| `smo2_from_running_max` | NIRS | 35,82         | 3,6 |
| `trainred_hhb_std` | NIRS | 32,86         | 3,3 |
| `z_vl_dist_rest_ratio_mid_high` | EMG | 29,00         | 2,9 |
| `z_vl_prox_load_e_d5` | EMG | 28,53         | 2,9 |
| `z_vl_dist_load_zcr` | EMG | 28,48         | 2,9 |

Источник: [results/final/shap/lt1_EN/shap_importance.csv](results/final/shap/lt1_EN/shap_importance.csv).

**Таблица А-2.2.** SHAP-важность по группам признаков для Ridge / EMG+NIRS / LT1.

| Группа | mean \|SHAP\| | Доля, % |
|--------|---------------|---------|
| EMG | 613,08        | 61,7 |
| NIRS | 334,69        | 33,7 |
| Кинематика | 45,80         | 4,6 |

Источник: [results/final/shap/lt1_EN/shap_by_group.csv](results/final/shap/lt1_EN/shap_by_group.csv).

## А-2.2 LT1 / EMG+NIRS+HRV (Ridge-проекция)

**Таблица А-2.3.** Топ-10 признаков мультимодальной модели Ridge / EMG+NIRS+HRV / LT1.

| Признак | Группа | mean \|SHAP\| | Доля, % |
|---------|--------|---------------|---------|
| `feat_rr_per_watt` | interaction | 46,46         | 4,9 |
| `hrv_mean_rr_ms` | HRV | 35,93         | 3,8 |
| `z_vl_dist_load_wavelet_entropy` | EMG | 35,82         | 3,8 |
| `trainred_hhb_mean` | NIRS | 35,75         | 3,8 |
| `trainred_smo2_drop` | NIRS | 34,78         | 3,7 |
| `z_vl_dist_load_zcr` | EMG | 33,73         | 3,5 |
| `trainred_hbdiff_std` | NIRS | 31,32         | 3,3 |
| `hhb_from_running_min` | NIRS | 30,00         | 3,2 |
| `hrv_dfa_alpha1` | HRV | 29,61         | 3,1 |
| `z_vl_prox_load_e_d5` | EMG | 23,22         | 2,4 |

Источник: [results/final/shap/lt1_ENH/shap_importance.csv](results/final/shap/lt1_ENH/shap_importance.csv).

**Таблица А-2.4.** SHAP-важность по группам признаков для Ridge / EMG+NIRS+HRV / LT1.

| Группа | mean \|SHAP\| | Доля, % |
|--------|---------------|---------|
| EMG | 504,45        | 53,1 |
| NIRS | 248,39        | 26,1 |
| HRV | 90,40         | 9,5 |
| Признаки взаимодействия | 72,37         | 7,6 |
| Кинематика | 34,83         | 3,7 |

Источник: [results/final/shap/lt1_ENH/shap_by_group.csv](results/final/shap/lt1_ENH/shap_by_group.csv).

## А-2.3 Сопоставление с SVR-PermutationExplainer

Сводное сопоставление по группам признаков для LT1 / EMG+NIRS:

| Группа | Ridge LinearExplainer | SVR PermutationExplainer | Δ, п.п. |
|--------|-----------------------|--------------------------|---------|
| EMG | 61,7 % | 54,0 % | −7,7 |
| NIRS | 33,7 % | 40,3 % | +6,6 |
| Кинематика | 4,6 % | 5,7 % | +1,1 |

Для LT1 / EMG+NIRS+HRV:

| Группа | Ridge | SVR | Δ, п.п. |
|--------|-------|-----|---------|
| EMG | 53,1 % | 46,1 % | −7,0 |
| NIRS | 26,1 % | 31,2 % | +5,1 |
| HRV | 9,5 % | 11,6 % | +2,1 |
| Признаки взаимодействия | 7,6 % | 6,8 % | −0,8 |
| Кинематика | 3,7 % | 4,4 % | +0,7 |

Содержательный сдвиг — увеличение доли NIRS на 5–7 п.п. при переходе от линейной проекции к SVR-разложению. Это объясняется тем, что NIRS-признаки описывают «изломы» мышечной оксигенации (drop, отклонения от running-экстремумов) и работают через нелинейные сочетания, не аппроксимируемые регуляризованной линейной регрессией. Иные группы изменяются слабее.

Качественный вывод (диффузность LT1 — нет доминирующего признака; основные источники сигнала — EMG > NIRS) сохраняется в обеих интерпретациях.

Топ-10 пересечений по признакам:

- LT1 / EMG+NIRS: Ridge-топ-10 и SVR-топ-10 содержат 6 общих признаков (`trainred_hhb_std`, `trainred_smo2_drop`, `smo2_from_running_max`, `hhb_from_running_min`, `trainred_hhb_mean`, `trainred_hbdiff_std`); 4 различающихся EMG-признака — в Ridge-топ-10 представлены, в SVR-топ-10 уступают восьми NIRS-признакам.
- LT1 / EMG+NIRS+HRV: 7 общих признаков, включая `hrv_mean_rr_ms`, `trainred_hhb_std`, `trainred_smo2_drop`, `hrv_dfa_alpha1`, `z_vl_dist_load_zcr`, `trainred_hhb_mean`, `smo2_from_running_max`.
