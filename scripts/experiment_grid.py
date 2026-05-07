"""Широкий grid-search baseline: 50+ конфигураций, LOSO CV, joblib параллелизм.

Что исследуем:
  - Наборы признаков: all, nirs_hrv, hrv_only, top-K по within-subj Spearman,
                      emg с per-subject нормировкой, stacked features
  - Модели: Ridge (4 α), ElasticNet, SVR (RBF, 3 C), ExtraTrees, RF (3 конфига),
            LightGBM (6 конфигов), Stacking [LightGBM + Ridge → Ridge]
  - Таргет: raw и sign-log компрессия
  - Предобработка: StandardScaler, RobustScaler, per-subject EMG z-score

Запуск:
  uv run python scripts/experiment_grid.py
"""

from __future__ import annotations

import sys
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import spearmanr
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.ensemble import (
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
    StackingRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.svm import SVR
import lightgbm as lgb

warnings.filterwarnings("ignore")

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

DATASET_PATH = _PROJECT_ROOT / "dataset" / "merged_features_ml.parquet"
RESULTS_DIR = _PROJECT_ROOT / "results" / "experiment_grid"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ─────────────────────── Столбцы по модальностям ───────────────────────

def get_modality_cols(df: pd.DataFrame) -> dict[str, list[str]]:
    emg = [c for c in df.columns if c.startswith("vl_") or c.startswith("delta_") or c.startswith("ratio_rms")]
    nirs = [c for c in df.columns if c.startswith("trainred_")]
    hrv = [c for c in df.columns if c.startswith("hrv_")
           and not c.endswith(("_valid", "_fraction", "_count"))]
    kin = [c for c in df.columns if c.startswith("cadence_") or c.startswith("load_") or c.startswith("rest_")]
    return {"emg": emg, "nirs": nirs, "hrv": hrv, "kin": kin}


# ─────────────────────── Кастомные трансформеры ───────────────────────

class SubjectEMGNormalizer(BaseEstimator, TransformerMixin):
    """Per-subject z-score только для EMG-признаков.

    Для каждого train-субъекта вычисляет mean/std и нормирует.
    Для test-субъектов применяет grand-mean/grand-std по всем train-субъектам.
    Работает в рамках одного LOSO-фолда: subject_ids передаются при fit.
    """

    def __init__(self, emg_indices: list[int]) -> None:
        self.emg_indices = emg_indices  # индексы EMG-колонок в X

    def fit(self, X: np.ndarray, y=None, subject_ids: np.ndarray | None = None) -> "SubjectEMGNormalizer":
        if subject_ids is None or len(self.emg_indices) == 0:
            self.grand_mean_ = np.zeros(len(self.emg_indices))
            self.grand_std_ = np.ones(len(self.emg_indices))
            return self
        # Нормируем каждый субъект, собираем распределение
        X_emg = X[:, self.emg_indices]
        subjs = np.unique(subject_ids)
        normed_chunks = []
        for s in subjs:
            mask = subject_ids == s
            chunk = X_emg[mask]
            mu = np.nanmean(chunk, axis=0)
            sigma = np.nanstd(chunk, axis=0)
            sigma = np.where(sigma < 1e-8, 1.0, sigma)
            normed_chunks.append((chunk - mu) / sigma)
        normed_all = np.vstack(normed_chunks)
        self.grand_mean_ = np.nanmean(normed_all, axis=0)
        self.grand_std_ = np.nanstd(normed_all, axis=0)
        self.grand_std_ = np.where(self.grand_std_ < 1e-8, 1.0, self.grand_std_)
        # Сохраняем per-subject stats для transform train-части
        self.subj_stats_: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for s in subjs:
            mask = subject_ids == s
            chunk = X_emg[mask]
            mu = np.nanmean(chunk, axis=0)
            sigma = np.nanstd(chunk, axis=0)
            sigma = np.where(sigma < 1e-8, 1.0, sigma)
            self.subj_stats_[s] = (mu, sigma)
        return self

    def transform(self, X: np.ndarray, subject_ids: np.ndarray | None = None) -> np.ndarray:
        X = X.copy().astype(float)
        if len(self.emg_indices) == 0:
            return X
        X_emg = X[:, self.emg_indices]
        if subject_ids is not None:
            for s in np.unique(subject_ids):
                mask = subject_ids == s
                if s in self.subj_stats_:
                    mu, sigma = self.subj_stats_[s]
                else:
                    # Тестовый субъект: grand stats
                    mu, sigma = self.grand_mean_, self.grand_std_
                X_emg[mask] = (X_emg[mask] - mu) / sigma
        X[:, self.emg_indices] = X_emg
        return X


class SignLogTargetTransformer:
    """sign(t) * log(|t| + offset) — сжимает хвосты таргета."""

    def __init__(self, offset: float = 60.0) -> None:
        self.offset = offset

    def transform(self, y: np.ndarray) -> np.ndarray:
        return np.sign(y) * np.log(np.abs(y) + self.offset)

    def inverse_transform(self, y: np.ndarray) -> np.ndarray:
        return np.sign(y) * (np.exp(np.abs(y)) - self.offset)


# ─────────────────────── Выбор топ-K признаков по Spearman ───────────────────────

def select_top_k_by_within_subject_spearman(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    k: int,
    subject_col: str = "subject_id",
) -> list[str]:
    """Выбирает top-K признаков по медиане |Spearman ρ| внутри субъектов."""
    rho_per_feat: dict[str, list[float]] = {c: [] for c in feature_cols}
    for subj in df[subject_col].unique():
        sub = df[df[subject_col] == subj]
        y_sub = sub[target_col].values
        for c in feature_cols:
            x = sub[c].values
            valid = np.isfinite(x) & np.isfinite(y_sub)
            if valid.sum() < 5:
                continue
            rho, _ = spearmanr(x[valid], y_sub[valid])
            if np.isfinite(rho):
                rho_per_feat[c].append(abs(rho))
    median_rho = {c: np.median(v) if v else 0.0 for c, v in rho_per_feat.items()}
    sorted_feats = sorted(median_rho, key=median_rho.get, reverse=True)
    return sorted_feats[:k]


# ─────────────────────── Метрики ───────────────────────

def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    mae = mean_absolute_error(y_true, y_pred)
    rmse = root_mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    rho, _ = spearmanr(y_true, y_pred)
    return {
        "mae_min": mae / 60.0,
        "rmse_min": rmse / 60.0,
        "r2": r2,
        "spearman_rho": rho,
    }


# ─────────────────────── Описание эксперимента ───────────────────────

@dataclass
class Experiment:
    """Одна конфигурация: набор признаков + модель + предобработка."""

    name: str
    feature_cols: list[str]
    make_pipeline: Callable[[], Pipeline]
    target_transform: SignLogTargetTransformer | None = None
    use_subj_emg_norm: bool = False
    emg_cols: list[str] = field(default_factory=list)


# ─────────────────────── LOSO для одного эксперимента ───────────────────────

def run_experiment_loso(
    exp: Experiment,
    df: pd.DataFrame,
    target_col: str,
) -> list[dict]:
    """Запускает LOSO для одной конфигурации. Возвращает список метрик по фолдам."""
    subjects = sorted(df["subject_id"].unique())
    records = []

    for subj in subjects:
        test_mask = df["subject_id"] == subj
        train_mask = ~test_mask

        X_train = df.loc[train_mask, exp.feature_cols].values
        y_train = df.loc[train_mask, target_col].values
        X_test = df.loc[test_mask, exp.feature_cols].values
        y_test = df.loc[test_mask, target_col].values

        if len(X_train) < 30 or len(X_test) == 0:
            continue

        # Per-subject EMG нормировка (до pipeline)
        if exp.use_subj_emg_norm and exp.emg_cols:
            emg_indices = [exp.feature_cols.index(c) for c in exp.emg_cols if c in exp.feature_cols]
            if emg_indices:
                norm = SubjectEMGNormalizer(emg_indices)
                train_subj_ids = df.loc[train_mask, "subject_id"].values
                norm.fit(X_train, subject_ids=train_subj_ids)
                X_train = norm.transform(X_train, subject_ids=train_subj_ids)
                X_test = norm.transform(X_test, subject_ids=np.array([subj] * len(X_test)))

        # Трансформация таргета
        if exp.target_transform is not None:
            y_train_fit = exp.target_transform.transform(y_train)
        else:
            y_train_fit = y_train

        pipe = clone(exp.make_pipeline())
        pipe.fit(X_train, y_train_fit)
        y_pred_transformed = pipe.predict(X_test)

        if exp.target_transform is not None:
            y_pred = exp.target_transform.inverse_transform(y_pred_transformed)
        else:
            y_pred = y_pred_transformed

        m = regression_metrics(y_test, y_pred)
        records.append({"subject_id": subj, "n_test": int(test_mask.sum()), **m})

    return records


def run_and_aggregate(
    exp: Experiment,
    df: pd.DataFrame,
    target_col: str,
) -> dict:
    """Запускает LOSO и агрегирует в одну строку результатов."""
    t0 = time.perf_counter()
    try:
        fold_records = run_experiment_loso(exp, df, target_col)
        if not fold_records:
            return {"name": exp.name, "error": "empty", "mae_min_mean": float("nan")}
        fold_df = pd.DataFrame(fold_records)
        elapsed = time.perf_counter() - t0
        return {
            "name": exp.name,
            "n_features": len(exp.feature_cols),
            "n_folds": len(fold_df),
            "elapsed_sec": round(elapsed, 1),
            "mae_min_mean": fold_df["mae_min"].mean(),
            "mae_min_std": fold_df["mae_min"].std(),
            "rmse_min_mean": fold_df["rmse_min"].mean(),
            "r2_mean": fold_df["r2"].mean(),
            "spearman_rho_mean": fold_df["spearman_rho"].mean(),
            "spearman_rho_std": fold_df["spearman_rho"].std(),
        }
    except Exception as exc:
        return {"name": exp.name, "error": str(exc)[:80], "mae_min_mean": float("nan")}


# ─────────────────────── Построение каталога экспериментов ───────────────────────

def build_experiment_catalog(df: pd.DataFrame, target_col: str) -> list[Experiment]:
    """Формирует полный список экспериментов."""
    cols = get_modality_cols(df)
    emg, nirs, hrv, kin = cols["emg"], cols["nirs"], cols["hrv"], cols["kin"]
    all_feats = emg + nirs + hrv + kin

    # Топ-K признаков по within-subject Spearman (вычисляем один раз)
    print("  Вычисляем within-subject Spearman для отбора признаков...", flush=True)
    top10 = select_top_k_by_within_subject_spearman(df, all_feats, target_col, k=10)
    top20 = select_top_k_by_within_subject_spearman(df, all_feats, target_col, k=20)
    top35 = select_top_k_by_within_subject_spearman(df, all_feats, target_col, k=35)
    print(f"  Top-10: {top10[:5]}...")

    # ─── Вспомогательные функции фабрики pipeline ───
    def std_pipeline(estimator):
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", estimator),
        ])

    def rob_pipeline(estimator):
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", RobustScaler()),
            ("model", estimator),
        ])

    def no_scale_pipeline(estimator):
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", estimator),
        ])

    # ─── Модели ───
    lgbm_default = lambda: lgb.LGBMRegressor(
        n_estimators=500, learning_rate=0.05, max_depth=6, num_leaves=31,
        min_child_samples=10, subsample=0.8, colsample_bytree=0.8,
        n_jobs=1, random_state=42, verbose=-1,
    )
    lgbm_deep = lambda: lgb.LGBMRegressor(
        n_estimators=800, learning_rate=0.03, max_depth=8, num_leaves=63,
        min_child_samples=8, subsample=0.7, colsample_bytree=0.7,
        n_jobs=1, random_state=42, verbose=-1,
    )
    lgbm_shallow = lambda: lgb.LGBMRegressor(
        n_estimators=300, learning_rate=0.1, max_depth=4, num_leaves=15,
        min_child_samples=15, subsample=0.9, colsample_bytree=0.9,
        n_jobs=1, random_state=42, verbose=-1,
    )
    lgbm_dart = lambda: lgb.LGBMRegressor(
        boosting_type="dart", n_estimators=400, learning_rate=0.05,
        max_depth=6, num_leaves=31, drop_rate=0.1,
        n_jobs=1, random_state=42, verbose=-1,
    )
    rf_200 = lambda: RandomForestRegressor(
        n_estimators=200, max_depth=8, min_samples_leaf=5, n_jobs=1, random_state=42,
    )
    rf_500_deep = lambda: RandomForestRegressor(
        n_estimators=500, max_depth=None, min_samples_leaf=3, n_jobs=1, random_state=42,
    )
    et_300 = lambda: ExtraTreesRegressor(
        n_estimators=300, max_depth=None, min_samples_leaf=3, n_jobs=1, random_state=42,
    )
    gbm = lambda: GradientBoostingRegressor(
        n_estimators=300, learning_rate=0.05, max_depth=5, subsample=0.8,
        min_samples_leaf=5, random_state=42,
    )
    svr_rbf_c1 = lambda: SVR(kernel="rbf", C=1.0, gamma="scale", epsilon=0.1)
    svr_rbf_c10 = lambda: SVR(kernel="rbf", C=10.0, gamma="scale", epsilon=0.1)
    svr_rbf_c100 = lambda: SVR(kernel="rbf", C=100.0, gamma="scale", epsilon=0.1)
    svr_poly = lambda: SVR(kernel="poly", C=10.0, degree=2, gamma="scale", epsilon=0.1)

    sign_log = SignLogTargetTransformer(offset=60.0)

    # ─── Stacking ───
    def stacking_lgbm_rf():
        estimators = [
            ("lgbm", lgbm_default()),
            ("rf", rf_200()),
        ]
        return StackingRegressor(
            estimators=estimators,
            final_estimator=Ridge(alpha=10.0),
            cv=5,
            n_jobs=1,
        )

    def stacking_lgbm_et_ridge():
        estimators = [
            ("lgbm", lgbm_default()),
            ("et", et_300()),
            ("ridge", Ridge(alpha=1.0)),
        ]
        return StackingRegressor(
            estimators=estimators,
            final_estimator=Ridge(alpha=1.0),
            cv=5,
            n_jobs=1,
        )

    experiments: list[Experiment] = []

    # ════════════════════════════════════════════════════════════════
    # БЛОК 1: Наборы признаков × Ridge (baseline регуляризации)
    # ════════════════════════════════════════════════════════════════
    for alpha in [0.01, 0.1, 1.0, 10.0, 100.0, 500.0]:
        for fs_name, feats in [("all", all_feats), ("nirs_hrv", nirs + hrv),
                                ("hrv_only", hrv), ("top10", top10),
                                ("top20", top20), ("top35", top35)]:
            a_tag = str(alpha).replace(".", "_")
            experiments.append(Experiment(
                name=f"Ridge_a{a_tag}_{fs_name}",
                feature_cols=feats,
                make_pipeline=lambda a=alpha: std_pipeline(Ridge(alpha=a)),
            ))

    # ════════════════════════════════════════════════════════════════
    # БЛОК 2: ElasticNet — отбор признаков через L1
    # ════════════════════════════════════════════════════════════════
    for alpha in [0.01, 0.1, 0.5]:
        for l1r in [0.1, 0.5, 0.9]:
            for fs_name, feats in [("all", all_feats), ("nirs_hrv", nirs + hrv)]:
                experiments.append(Experiment(
                    name=f"ElasticNet_a{alpha}_l1r{l1r}_{fs_name}",
                    feature_cols=feats,
                    make_pipeline=lambda a=alpha, l=l1r: std_pipeline(
                        ElasticNet(alpha=a, l1_ratio=l, max_iter=2000)
                    ),
                ))

    # ════════════════════════════════════════════════════════════════
    # БЛОК 3: SVR (малые наборы признаков — иначе слишком медленно)
    # ════════════════════════════════════════════════════════════════
    for fs_name, feats in [("nirs_hrv", nirs + hrv), ("hrv_only", hrv),
                            ("top10", top10), ("top20", top20)]:
        for svr_name, svr_fn in [("rbf_C1", svr_rbf_c1), ("rbf_C10", svr_rbf_c10),
                                  ("rbf_C100", svr_rbf_c100), ("poly_C10", svr_poly)]:
            experiments.append(Experiment(
                name=f"SVR_{svr_name}_{fs_name}",
                feature_cols=feats,
                make_pipeline=lambda fn=svr_fn: std_pipeline(fn()),
            ))
        # RobustScaler вариант
        experiments.append(Experiment(
            name=f"SVR_rbf_C10_rob_{fs_name}",
            feature_cols=feats,
            make_pipeline=lambda: rob_pipeline(svr_rbf_c10()),
        ))

    # ════════════════════════════════════════════════════════════════
    # БЛОК 4: RandomForest / ExtraTrees / GBM
    # ════════════════════════════════════════════════════════════════
    for fs_name, feats in [("all", all_feats), ("nirs_hrv", nirs + hrv),
                            ("hrv_only", hrv), ("top20", top20), ("top35", top35)]:
        experiments.append(Experiment(
            name=f"RF200_{fs_name}",
            feature_cols=feats,
            make_pipeline=lambda fn=rf_200: no_scale_pipeline(fn()),
        ))
        experiments.append(Experiment(
            name=f"RF500deep_{fs_name}",
            feature_cols=feats,
            make_pipeline=lambda fn=rf_500_deep: no_scale_pipeline(fn()),
        ))
        experiments.append(Experiment(
            name=f"ET300_{fs_name}",
            feature_cols=feats,
            make_pipeline=lambda fn=et_300: no_scale_pipeline(fn()),
        ))
        experiments.append(Experiment(
            name=f"GBM300_{fs_name}",
            feature_cols=feats,
            make_pipeline=lambda fn=gbm: no_scale_pipeline(fn()),
        ))

    # ════════════════════════════════════════════════════════════════
    # БЛОК 5: LightGBM — разные конфиги × наборы признаков
    # ════════════════════════════════════════════════════════════════
    lgbm_variants = [
        ("default", lgbm_default),
        ("deep", lgbm_deep),
        ("shallow", lgbm_shallow),
        ("dart", lgbm_dart),
    ]
    for fs_name, feats in [("all", all_feats), ("nirs_hrv", nirs + hrv),
                            ("hrv_only", hrv), ("top20", top20), ("top35", top35),
                            ("emg_nirs", emg + nirs)]:
        for lgbm_tag, lgbm_fn in lgbm_variants:
            experiments.append(Experiment(
                name=f"LGBM_{lgbm_tag}_{fs_name}",
                feature_cols=feats,
                make_pipeline=lambda fn=lgbm_fn: no_scale_pipeline(fn()),
            ))

    # ════════════════════════════════════════════════════════════════
    # БЛОК 6: Sign-log трансформация таргета
    # ════════════════════════════════════════════════════════════════
    for fs_name, feats in [("nirs_hrv", nirs + hrv), ("hrv_only", hrv), ("top20", top20)]:
        for lgbm_tag, lgbm_fn in [("default", lgbm_default), ("deep", lgbm_deep)]:
            experiments.append(Experiment(
                name=f"LGBM_{lgbm_tag}_{fs_name}_signlog",
                feature_cols=feats,
                make_pipeline=lambda fn=lgbm_fn: no_scale_pipeline(fn()),
                target_transform=SignLogTargetTransformer(60.0),
            ))
        experiments.append(Experiment(
            name=f"Ridge_a1_{fs_name}_signlog",
            feature_cols=feats,
            make_pipeline=lambda: std_pipeline(Ridge(alpha=1.0)),
            target_transform=SignLogTargetTransformer(60.0),
        ))

    # ════════════════════════════════════════════════════════════════
    # БЛОК 7: Per-subject EMG нормировка (ключевой эксперимент)
    # ════════════════════════════════════════════════════════════════
    for fs_name, feats in [("all", all_feats), ("emg_nirs", emg + nirs), ("all_kin", all_feats)]:
        for lgbm_tag, lgbm_fn in [("default", lgbm_default), ("deep", lgbm_deep)]:
            experiments.append(Experiment(
                name=f"LGBM_{lgbm_tag}_{fs_name}_subjEMGnorm",
                feature_cols=feats,
                make_pipeline=lambda fn=lgbm_fn: no_scale_pipeline(fn()),
                use_subj_emg_norm=True,
                emg_cols=emg,
            ))
        experiments.append(Experiment(
            name=f"Ridge_a10_{fs_name}_subjEMGnorm",
            feature_cols=feats,
            make_pipeline=lambda: std_pipeline(Ridge(alpha=10.0)),
            use_subj_emg_norm=True,
            emg_cols=emg,
        ))
        experiments.append(Experiment(
            name=f"SVR_rbf_C10_{fs_name}_subjEMGnorm",
            feature_cols=feats,
            make_pipeline=lambda: std_pipeline(svr_rbf_c10()),
            use_subj_emg_norm=True,
            emg_cols=emg,
        ))

    # ════════════════════════════════════════════════════════════════
    # БЛОК 8: Stacking
    # ════════════════════════════════════════════════════════════════
    for fs_name, feats in [("nirs_hrv", nirs + hrv), ("hrv_only", hrv),
                            ("top20", top20), ("all", all_feats)]:
        experiments.append(Experiment(
            name=f"Stack_LGBM_RF_{fs_name}",
            feature_cols=feats,
            make_pipeline=lambda fn=stacking_lgbm_rf: no_scale_pipeline(fn()),
        ))
    experiments.append(Experiment(
        name="Stack_LGBM_ET_Ridge_top20",
        feature_cols=top20,
        make_pipeline=lambda: no_scale_pipeline(stacking_lgbm_et_ridge()),
    ))

    # ════════════════════════════════════════════════════════════════
    # БЛОК 9: RobustScaler для LGBM и Ridge (контрольный)
    # ════════════════════════════════════════════════════════════════
    for fs_name, feats in [("nirs_hrv", nirs + hrv), ("top20", top20)]:
        experiments.append(Experiment(
            name=f"Ridge_a1_rob_{fs_name}",
            feature_cols=feats,
            make_pipeline=lambda: rob_pipeline(Ridge(alpha=1.0)),
        ))

    return experiments


# ─────────────────────── main ───────────────────────

def main() -> None:
    print("=" * 72)
    print("EXPERIMENT GRID — LOSO CV")
    print("=" * 72)

    df = pd.read_parquet(DATASET_PATH)
    df_valid = df[df["window_valid_all_required"] == 1].copy()
    TARGET = "target_time_to_lt2_center_sec"
    df_valid = df_valid.dropna(subset=[TARGET])

    print(f"Датасет: {len(df_valid)} окон, {df_valid['subject_id'].nunique()} участников")

    # Каталог экспериментов
    experiments = build_experiment_catalog(df_valid, TARGET)
    print(f"\nВсего конфигураций: {len(experiments)}")
    print("Запускаем параллельно (n_jobs=-1)...\n")

    t_start = time.perf_counter()

    # Параллельный запуск — каждый эксперимент получает свой LOSO
    results_raw = Parallel(n_jobs=-1, verbose=5)(
        delayed(run_and_aggregate)(exp, df_valid, TARGET)
        for exp in experiments
    )

    elapsed_total = time.perf_counter() - t_start
    print(f"\nВсего времени: {elapsed_total:.1f} с")

    # ─── Сборка результатов ───
    results_df = pd.DataFrame(results_raw)
    # Убираем упавшие
    errors = results_df[results_df.get("error", pd.Series(dtype=str)).notna() & results_df["mae_min_mean"].isna()]
    if len(errors):
        print(f"\n⚠️  Упало экспериментов: {len(errors)}")
        print(errors[["name", "error"]].to_string(index=False))

    results_df = results_df.dropna(subset=["mae_min_mean"])
    results_df = results_df.sort_values("mae_min_mean").reset_index(drop=True)
    results_df.insert(0, "rank", range(1, len(results_df) + 1))

    # ─── Сохранение ───
    out_csv = RESULTS_DIR / "leaderboard.csv"
    results_df.to_csv(out_csv, index=False)
    print(f"\nЛидерборд сохранён: {out_csv}")

    # ─── Топ-20 ───
    pd.set_option("display.max_colwidth", 45)
    pd.set_option("display.float_format", "{:.3f}".format)
    print("\n" + "═" * 90)
    print("ТОП-20 КОНФИГУРАЦИЙ по MAE (мин)")
    print("═" * 90)
    show_cols = ["rank", "name", "n_features", "mae_min_mean", "mae_min_std",
                 "r2_mean", "spearman_rho_mean", "elapsed_sec"]
    print(results_df.head(20)[show_cols].to_string(index=False))

    # ─── Анализ по группам ───
    print("\n" + "═" * 90)
    print("ЛУЧШИЙ РЕЗУЛЬТАТ ПО ГРУППАМ МОДЕЛЕЙ")
    print("═" * 90)
    for prefix in ["Ridge", "ElasticNet", "SVR", "RF", "ET", "GBM", "LGBM", "Stack"]:
        subset = results_df[results_df["name"].str.startswith(prefix)]
        if len(subset):
            row = subset.iloc[0]
            print(f"  {prefix:12s}: {row['name']:<45s} MAE={row['mae_min_mean']:.3f}±{row['mae_min_std']:.3f}")

    print("\n" + "═" * 90)
    print("ЛУЧШИЕ ПО НАБОРАМ ПРИЗНАКОВ (LGBM)")
    print("═" * 90)
    lgbm_res = results_df[results_df["name"].str.startswith("LGBM")]
    for tag in ["hrv_only", "nirs_hrv", "all", "emg_nirs", "top20", "top10", "signlog", "subjEMGnorm"]:
        subset = lgbm_res[lgbm_res["name"].str.contains(tag)]
        if len(subset):
            row = subset.iloc[0]
            print(f"  {tag:20s}: {row['name']:<45s} MAE={row['mae_min_mean']:.3f}±{row['mae_min_std']:.3f}")

    print(f"\n✅  Готово за {elapsed_total:.0f} с. Результаты: {RESULTS_DIR.resolve()}")


if __name__ == "__main__":
    main()
