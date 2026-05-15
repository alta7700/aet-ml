# ENV — окружение расчётов results/final/

- Дата сборки: 2026-05-15
- Python: 3.13.5
- Платформа: macOS-26.2-arm64-arm-64bit-Mach-O

## Версии пакетов

| Пакет | Версия |
|-------|--------|
| numpy | 2.4.4 |
| pandas | 3.0.2 |
| scikit-learn | 1.8.0 |
| scipy | 1.17.1 |
| shap | 0.51.0 |
| matplotlib | 3.10.9 |
| pyarrow | 24.0.0 |
| h5py | 3.16.0 |

## Сиды и детерминизм

- LOSO детерминирован (нет случайности в Ridge/sklearn-зоопарке).
- Маршрутизатор (LogisticRegression) — детерминирован при фиксированных данных.
- Нейросетевые версии (v01xx) обучались на GPU-сервере; их предсказания берутся из сохранённых npy, не пересчитываются здесь.
- Калибровка conformal — LOSO split-conformal, без случайности.

## Воспроизведение

```bash
# порядок запуска скриптов шагов 01–07
uv run python scripts/patch_session_params_hrv_baseline.py   # шаг 01
PYTHONPATH=. uv run python scripts/final_per_subject_from_npy.py  # шаг 02 (данные)
PYTHONPATH=. uv run python scripts/final_build_ranking.py        # шаг 02
PYTHONPATH=. uv run python scripts/final_topk_views.py           # шаг 02
PYTHONPATH=. uv run python scripts/final_per_subject_dist.py     # шаг 02a
PYTHONPATH=. uv run python scripts/final_time_resolved.py        # шаг 02b
PYTHONPATH=. uv run python scripts/final_abs_vs_noabs.py         # шаг 03
PYTHONPATH=. uv run python scripts/final_v0011_hrv_coefs.py      # шаг 03 (аддендум)
PYTHONPATH=. uv run python scripts/final_pairwise_wilcoxon.py    # шаг 04
PYTHONPATH=. uv run python scripts/final_two_experts.py          # шаг 05
PYTHONPATH=. uv run python scripts/final_shap_conformal.py       # шаг 06
PYTHONPATH=. uv run python scripts/final_package.py              # шаг 07
```