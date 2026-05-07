"""Baseline модели ML для предсказания времени до LT2.

Запуск:
  uv run python scripts/train_baseline.py

Стратегия оценки:
  Leave-One-Subject-Out (LOSO) — 13 фолдов, каждый участник по одному разу в тесте.
  Обучение строго на тренировочных субъектах (нет data leakage).

Предобработка (fit только на train-фолде):
  1. Median imputer — заполнение NaN медианой по обучению
  2. StandardScaler — стандартизация

Задачи:
  A) Регрессия: target_time_to_lt2_center_sec (в секундах)
  B) Классификация: target_binary_label (0=до LT2, 1=после LT2)

Модели:
  - Ridge (λ=1.0)
  - RandomForest (200 деревьев)
  - LightGBM (500 деревьев)

Наборы признаков:
  - all: все 99 признаков (EMG + NIRS + HRV + кинематика)
  - nirs_hrv: только NIRS + HRV (15 + 7 = 22 признака)
  - emg_nirs: только EMG + NIRS (72 + 15 = 87 признаков)
"""

from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    r2_score,
    roc_auc_score,
    root_mean_squared_error,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from scipy.stats import spearmanr

import lightgbm as lgb

warnings.filterwarnings("ignore")

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# ─────────────────────── Конфигурация ───────────────────────

DATASET_PATH = _PROJECT_ROOT / "dataset" / "merged_features_ml.parquet"
RESULTS_DIR = _PROJECT_ROOT / "results" / "baseline"

# Минимальная доля обучающих окон на фолд (защита от пустых фолдов)
MIN_TRAIN_SAMPLES = 50


# ─────────────────────── Признаки ───────────────────────

def get_feature_sets(df: pd.DataFrame) -> dict[str, list[str]]:
    """Возвращает словарь наборов признаков {имя → список колонок}."""
    emg = [c for c in df.columns if c.startswith("vl_") or c.startswith("delta_") or c.startswith("ratio_rms")]
    nirs = [c for c in df.columns if c.startswith("trainred_")]
    hrv = [
        c for c in df.columns
        if c.startswith("hrv_")
        and not c.endswith("_valid")
        and not c.endswith("_fraction")
        and not c.endswith("_count")
    ]
    kin = [c for c in df.columns if c.startswith("cadence_") or c.startswith("load_") or c.startswith("rest_")]

    return {
        "all":      emg + nirs + hrv + kin,
        "nirs_hrv": nirs + hrv,
        "emg_nirs": emg + nirs,
        "hrv_only": hrv,
    }


# ─────────────────────── Метрики ───────────────────────

def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Вычисляет метрики регрессии."""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = root_mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    rho, _ = spearmanr(y_true, y_pred)
    # Переводим в минуты для читаемости
    mae_min = mae / 60.0
    rmse_min = rmse / 60.0
    return {
        "mae_sec": mae,
        "mae_min": mae_min,
        "rmse_sec": rmse,
        "rmse_min": rmse_min,
        "r2": r2,
        "spearman_rho": rho,
    }


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray | None) -> dict[str, float]:
    """Вычисляет метрики бинарной классификации."""
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    auc = roc_auc_score(y_true, y_proba) if y_proba is not None and len(np.unique(y_true)) == 2 else float("nan")
    return {"accuracy": acc, "f1": f1, "roc_auc": auc}


# ─────────────────────── Построение Pipeline ───────────────────────

def make_regression_pipelines() -> dict[str, Pipeline]:
    """Создаёт пайплайны регрессии {имя → Pipeline}."""
    preproc = [
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ]
    return {
        "Ridge": Pipeline(preproc + [("model", Ridge(alpha=1.0))]),
        "RandomForest": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", RandomForestRegressor(
                n_estimators=200,
                max_depth=8,
                min_samples_leaf=5,
                n_jobs=-1,
                random_state=42,
            )),
        ]),
        "LightGBM": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", lgb.LGBMRegressor(
                n_estimators=500,
                learning_rate=0.05,
                max_depth=6,
                num_leaves=31,
                min_child_samples=10,
                subsample=0.8,
                colsample_bytree=0.8,
                n_jobs=-1,
                random_state=42,
                verbose=-1,
            )),
        ]),
    }


def make_classification_pipelines() -> dict[str, Pipeline]:
    """Создаёт пайплайны классификации {имя → Pipeline}."""
    preproc = [
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ]
    return {
        "LogReg": Pipeline(preproc + [("model", LogisticRegression(C=1.0, max_iter=1000, random_state=42))]),
        "RandomForest": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", RandomForestClassifier(
                n_estimators=200,
                max_depth=8,
                min_samples_leaf=5,
                n_jobs=-1,
                random_state=42,
            )),
        ]),
        "LightGBM": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", lgb.LGBMClassifier(
                n_estimators=500,
                learning_rate=0.05,
                max_depth=6,
                num_leaves=31,
                min_child_samples=10,
                subsample=0.8,
                colsample_bytree=0.8,
                n_jobs=-1,
                random_state=42,
                verbose=-1,
            )),
        ]),
    }


# ─────────────────────── LOSO CV ───────────────────────

def run_loso_regression(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    pipelines: dict[str, Pipeline],
) -> pd.DataFrame:
    """Запускает LOSO CV для регрессии.

    Параметры
    ----------
    df : DataFrame
        Только валидные окна с не-NaN таргетом.
    feature_cols : list[str]
        Список признаков.
    target_col : str
        Название колонки таргета.
    pipelines : dict
        {имя_модели → Pipeline}.

    Возвращает
    ----------
    DataFrame с метриками по каждому фолду и модели.
    """
    subjects = sorted(df["subject_id"].unique())
    records = []

    for subj in subjects:
        test_mask = df["subject_id"] == subj
        train_mask = ~test_mask

        X_train = df.loc[train_mask, feature_cols].values
        y_train = df.loc[train_mask, target_col].values
        X_test = df.loc[test_mask, feature_cols].values
        y_test = df.loc[test_mask, target_col].values

        if len(X_train) < MIN_TRAIN_SAMPLES or len(X_test) == 0:
            print(f"    ⚠️  {subj}: мало данных (train={len(X_train)}, test={len(X_test)}), пропускаем")
            continue

        for model_name, pipe in pipelines.items():
            t0 = time.perf_counter()
            pipe_clone = type(pipe)(pipe.steps)  # клонирование не нужно — fit всегда перезаписывает
            # scikit-learn clone
            from sklearn.base import clone
            pipe_clone = clone(pipe)
            pipe_clone.fit(X_train, y_train)
            y_pred = pipe_clone.predict(X_test)
            elapsed = time.perf_counter() - t0

            m = regression_metrics(y_test, y_pred)
            records.append({
                "subject_id": subj,
                "model": model_name,
                "n_test": len(y_test),
                "elapsed_sec": round(elapsed, 2),
                **m,
            })

    return pd.DataFrame(records)


def run_loso_classification(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    pipelines: dict[str, Pipeline],
) -> pd.DataFrame:
    """Запускает LOSO CV для классификации."""
    subjects = sorted(df["subject_id"].unique())
    records = []

    for subj in subjects:
        test_mask = df["subject_id"] == subj
        train_mask = ~test_mask

        X_train = df.loc[train_mask, feature_cols].values
        y_train = df.loc[train_mask, target_col].values
        X_test = df.loc[test_mask, feature_cols].values
        y_test = df.loc[test_mask, target_col].values

        if len(X_train) < MIN_TRAIN_SAMPLES or len(X_test) == 0:
            continue

        # Проверяем наличие обоих классов в тренировке
        if len(np.unique(y_train)) < 2:
            print(f"    ⚠️  {subj}: только 1 класс в train, пропускаем классификацию")
            continue

        for model_name, pipe in pipelines.items():
            from sklearn.base import clone
            t0 = time.perf_counter()
            pipe_clone = clone(pipe)
            pipe_clone.fit(X_train, y_train)
            y_pred = pipe_clone.predict(X_test)

            # Вероятности для AUC
            y_proba = None
            if hasattr(pipe_clone[-1], "predict_proba"):
                y_proba = pipe_clone.predict_proba(X_test)[:, 1]
            elapsed = time.perf_counter() - t0

            m = classification_metrics(y_test, y_pred, y_proba)
            records.append({
                "subject_id": subj,
                "model": model_name,
                "n_test": len(y_test),
                "n_pos_test": int(y_test.sum()),
                "elapsed_sec": round(elapsed, 2),
                **m,
            })

    return pd.DataFrame(records)


# ─────────────────────── Отчёт ───────────────────────

def print_regression_summary(results: pd.DataFrame, title: str) -> None:
    """Печатает сводную таблицу метрик регрессии."""
    print(f"\n{'─' * 70}")
    print(f"РЕГРЕССИЯ: {title}")
    print(f"{'─' * 70}")
    summary = (
        results.groupby("model")[["mae_min", "rmse_min", "r2", "spearman_rho"]]
        .agg(["mean", "std"])
        .round(3)
    )
    print(summary.to_string())

    print(f"\nЛучшая модель по MAE (мин):")
    best = results.groupby("model")["mae_min"].mean().idxmin()
    mae_mean = results.groupby("model")["mae_min"].mean()[best]
    mae_std = results.groupby("model")["mae_min"].std()[best]
    print(f"  {best}: MAE = {mae_mean:.2f} ± {mae_std:.2f} мин")


def print_classification_summary(results: pd.DataFrame, title: str) -> None:
    """Печатает сводную таблицу метрик классификации."""
    print(f"\n{'─' * 70}")
    print(f"КЛАССИФИКАЦИЯ: {title}")
    print(f"{'─' * 70}")
    summary = (
        results.groupby("model")[["accuracy", "f1", "roc_auc"]]
        .agg(["mean", "std"])
        .round(3)
    )
    print(summary.to_string())

    print(f"\nЛучшая модель по ROC-AUC:")
    best = results.groupby("model")["roc_auc"].mean().idxmax()
    auc_mean = results.groupby("model")["roc_auc"].mean()[best]
    auc_std = results.groupby("model")["roc_auc"].std()[best]
    print(f"  {best}: AUC = {auc_mean:.3f} ± {auc_std:.3f}")


def save_results(results: pd.DataFrame, name: str, out_dir: Path) -> None:
    """Сохраняет результаты в CSV."""
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.csv"
    results.to_csv(path, index=False)
    print(f"  → Сохранено: {path}")


# ─────────────────────── main ───────────────────────

def main() -> None:
    """Точка входа."""
    print("=" * 70)
    print("BASELINE MODELS — LOSO CV")
    print("=" * 70)

    # ── Загрузка данных ──
    df = pd.read_parquet(DATASET_PATH)
    print(f"\nДатасет: {len(df)} окон, {df['subject_id'].nunique()} участников")

    feature_sets = get_feature_sets(df)
    for name, cols in feature_sets.items():
        print(f"  Набор '{name}': {len(cols)} признаков")

    # ── Фильтрация: только все-3-модальности валидны ──
    df_valid = df[df["window_valid_all_required"] == 1].copy()
    print(f"\nПосле фильтра (все 3 модальности): {len(df_valid)} окон")
    print(f"Субъекты: {sorted(df_valid['subject_id'].unique())}")

    # ── Задача A: Регрессия ──
    REG_TARGET = "target_time_to_lt2_center_sec"
    df_reg = df_valid[[REG_TARGET] + ["subject_id"] + sum(feature_sets.values(), [])].copy()
    df_reg = df_reg.dropna(subset=[REG_TARGET])
    print(f"\n[Регрессия] Окон с таргетом: {len(df_reg)}")

    reg_pipelines = make_regression_pipelines()

    # Только 'all' и 'nirs_hrv' для скорости на первом прогоне
    for fs_name in ["all", "nirs_hrv", "emg_nirs", "hrv_only"]:
        feat_cols = feature_sets[fs_name]
        # Оставляем только колонки, реально присутствующие
        feat_cols = [c for c in feat_cols if c in df_reg.columns]
        print(f"\n  [Набор: {fs_name}, {len(feat_cols)} признаков]")
        t0 = time.perf_counter()
        res = run_loso_regression(df_reg, feat_cols, REG_TARGET, reg_pipelines)
        elapsed = time.perf_counter() - t0
        print(f"  Время: {elapsed:.1f} с")
        print_regression_summary(res, fs_name)
        save_results(res, f"reg_{fs_name}", RESULTS_DIR)

    # ── Задача B: Классификация ──
    CLS_TARGET = "target_binary_label"
    df_cls = df_valid.copy()
    # target_binary_label: NaN для окон в «серой зоне» LT2 — исключаем
    df_cls = df_cls[df_cls[CLS_TARGET].notna()].copy()
    df_cls[CLS_TARGET] = df_cls[CLS_TARGET].astype(int)
    print(f"\n[Классификация] Окон: {len(df_cls)}, класс 1 (после LT2): {int(df_cls[CLS_TARGET].sum())}")

    cls_pipelines = make_classification_pipelines()

    for fs_name in ["all", "nirs_hrv", "emg_nirs", "hrv_only"]:
        feat_cols = feature_sets[fs_name]
        feat_cols = [c for c in feat_cols if c in df_cls.columns]
        print(f"\n  [Набор: {fs_name}, {len(feat_cols)} признаков]")
        t0 = time.perf_counter()
        res = run_loso_classification(df_cls, feat_cols, CLS_TARGET, cls_pipelines)
        elapsed = time.perf_counter() - t0
        print(f"  Время: {elapsed:.1f} с")
        print_classification_summary(res, fs_name)
        save_results(res, f"cls_{fs_name}", RESULTS_DIR)

    print(f"\n{'=' * 70}")
    print(f"✅ Готово. Результаты: {RESULTS_DIR.resolve()}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
