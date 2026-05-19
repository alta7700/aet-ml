"""SHAP-анализ для семейства классических моделей.

Подход:
  1. Берём автоматически выбранные finalist-модели из selection-layer.
  2. Для каждого LOSO-fold загружаем сохранённый sklearn pipeline
     (imputer + scaler + model) из joblib-чекпоинта.
  3. Считаем SHAP для оконного регрессионного выхода.
  4. Агрегируем важности до уровня субъекта, модели и модальности.

SHAP считается только для family="Lin" в рамках текущего selection scope.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from common_lib import build_fold_id
from dataset_pipeline.common import DEFAULT_DATASET_DIR
from features import (
    HRV_FEATURES,
    INTERACTION_FEATURES,
    NIRS_FEATURES,
    RUNNING_NIRS_FEATURES,
    get_feature_cols,
    prepare_data,
)

from analysis.selection import build_selection_tables
from analysis.schemas import AnalysisConfig

TARGET_COLS = {
    "lt1": "target_time_to_lt1_pchip_sec",
    "lt2": "target_time_to_lt2_center_sec",
}


def _sample_rows(df: pd.DataFrame, n: int, rng: np.random.Generator) -> pd.DataFrame:
    """Возвращает детерминированную подвыборку строк без замены."""
    if len(df) <= n:
        return df.copy()
    seed = int(rng.integers(0, 2**31 - 1))
    return df.sample(n=n, replace=False, random_state=seed)


def _feature_modality(feature_name: str) -> str:
    """Грубое отображение признака в модальность для агрегированных сводок."""
    if feature_name in INTERACTION_FEATURES:
        return "Interaction"
    if feature_name in HRV_FEATURES:
        return "HRV"
    if feature_name in NIRS_FEATURES or feature_name in RUNNING_NIRS_FEATURES:
        return "NIRS"
    if feature_name.startswith("z_vl_"):
        return "EMG"
    if feature_name.startswith("z_"):
        return "Kinematics"
    return "Other"


def _prepare_target_frames() -> dict[str, pd.DataFrame]:
    """Готовит target-specific dataframes и добавляет sample_index per subject."""
    dataset_path = DEFAULT_DATASET_DIR / "merged_features_ml.parquet"
    session_params_path = DEFAULT_DATASET_DIR / "session_params.parquet"
    raw = pd.read_parquet(dataset_path)
    session_params = (pd.read_parquet(session_params_path)
                      if session_params_path.exists() else pd.DataFrame())

    out: dict[str, pd.DataFrame] = {}
    for target, target_col in TARGET_COLS.items():
        df = prepare_data(raw, session_params, target)
        df = df.dropna(subset=[target_col]).sort_values(
            ["subject_id", "window_start_sec"]).reset_index(drop=True)
        df["sample_index"] = df.groupby("subject_id").cumcount().astype(np.int64)
        out[target] = df
    return out


def _checkpoint_path(cfg: AnalysisConfig, row: pd.Series) -> Path:
    """Возвращает путь к grouped joblib-чекпоинту одной Lin-модели."""
    return (cfg.results_root / str(row["architecture_id"]) / str(row["model_id"]) /
            f"model_{row['model_id']}_epoch-000.joblib")


def _compute_shap_values(model, x_bg: np.ndarray, x_eval: np.ndarray,
                         cfg: AnalysisConfig) -> tuple[np.ndarray, np.ndarray]:
    """Считает SHAP-значения для одной обученной sklearn-модели."""
    if hasattr(model, "estimators_"):
        explainer = shap.TreeExplainer(model)
        explanation = explainer(x_eval)
        return np.asarray(explanation.values), np.asarray(explanation.base_values)

    if hasattr(model, "coef_"):
        explainer = shap.LinearExplainer(model, x_bg)
        explanation = explainer(x_eval)
        return np.asarray(explanation.values), np.asarray(explanation.base_values)

    bg_small = x_bg
    if len(bg_small) > cfg.shap.kernel_background_samples:
        bg_small = bg_small[:cfg.shap.kernel_background_samples]
    explainer = shap.KernelExplainer(model.predict, bg_small)
    shap_values = explainer.shap_values(
        x_eval,
        nsamples=cfg.shap.kernel_nsamples,
        silent=True,
    )
    shap_values = np.asarray(shap_values)
    base_values = np.repeat(float(np.asarray(explainer.expected_value).reshape(-1)[0]),
                            len(x_eval))
    return shap_values, base_values


def _aggregate_subject_feature_rows(*, model_row: pd.Series, subject_id: str,
                                    feature_names: list[str],
                                    x_eval_raw: pd.DataFrame,
                                    y_pred: np.ndarray,
                                    base_values: np.ndarray,
                                    shap_values: np.ndarray) -> list[dict[str, object]]:
    """Сворачивает оконный SHAP в subject-level таблицу по признакам."""
    rows: list[dict[str, object]] = []
    n_windows = int(len(x_eval_raw))
    mean_abs = np.mean(np.abs(shap_values), axis=0)
    mean_signed = np.mean(shap_values, axis=0)
    mean_feature_value = np.mean(x_eval_raw.to_numpy(dtype=float), axis=0)
    mean_pred = float(np.mean(y_pred))
    mean_base = float(np.mean(base_values))

    for idx, feature_name in enumerate(feature_names):
        rows.append({
            "model_id": model_row["model_id"],
            "architecture_id": model_row["architecture_id"],
            "family": model_row["family"],
            "target": model_row["target"],
            "feature_set": model_row["feature_set"],
            "with_abs": bool(model_row["with_abs"]),
            "phase_split": model_row["phase_split"],
            "subject_id": subject_id,
            "feature_name": feature_name,
            "modality": _feature_modality(feature_name),
            "mean_abs_shap": float(mean_abs[idx]),
            "mean_shap": float(mean_signed[idx]),
            "mean_feature_value": float(mean_feature_value[idx]),
            "mean_prediction": mean_pred,
            "mean_base_value": mean_base,
            "n_windows": n_windows,
        })
    return rows


def _normalise_share(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    """Добавляет относительную долю |SHAP| внутри группы."""
    out = df.copy()
    denom = out.groupby(group_cols)["mean_abs_shap"].transform("sum")
    out["abs_shap_share"] = out["mean_abs_shap"] / (denom + 1e-12)
    return out


def build_shap_tables(model_summary: pd.DataFrame,
                      preds_selected: pd.DataFrame,
                      cfg: AnalysisConfig) -> dict[str, pd.DataFrame]:
    """Считает SHAP-таблицы для finalist-моделей семейства Lin."""
    if not cfg.shap.enabled:
        empty = pd.DataFrame()
        for path in (
            cfg.shap_global_summary_path,
            cfg.shap_subject_summary_path,
            cfg.shap_modality_summary_path,
        ):
            path.parent.mkdir(parents=True, exist_ok=True)
            empty.to_parquet(path, index=False)
        return {"global": empty, "subject": empty, "modality": empty}

    selected = build_selection_tables(model_summary, cfg)["selected_for_shap"].copy()
    if selected.empty:
        empty = pd.DataFrame()
        for path in (
            cfg.shap_global_summary_path,
            cfg.shap_subject_summary_path,
            cfg.shap_modality_summary_path,
        ):
            path.parent.mkdir(parents=True, exist_ok=True)
            empty.to_parquet(path, index=False)
        return {"global": empty, "subject": empty, "modality": empty}

    rng = np.random.default_rng(cfg.shap.random_seed)
    prepared = _prepare_target_frames()
    preds_use = preds_selected[preds_selected["model_id"].isin(selected["model_id"])].copy()

    subject_rows: list[dict[str, object]] = []
    for _, model_row in selected.iterrows():
        target = str(model_row["target"])
        df_target = prepared[target]
        feat_cols = get_feature_cols(
            df_target,
            str(model_row["feature_set"]),
            with_abs=bool(model_row["with_abs"]),
            phase_split=str(model_row["phase_split"]),
        )
        feat_frame = df_target[["subject_id", "sample_index", *feat_cols]].copy()

        ckpt = _checkpoint_path(cfg, model_row)
        if not ckpt.exists():
            raise FileNotFoundError(f"Не найден checkpoint для SHAP: {ckpt}")
        fold_states = joblib.load(ckpt)

        preds_model = preds_use[preds_use["model_id"] == model_row["model_id"]].copy()
        for subject_id, pred_sub in preds_model.groupby("subject_id", sort=True):
            fold_id = build_fold_id(subject_id)
            if fold_id not in fold_states:
                raise KeyError(
                    f"{model_row['model_id']}: нет fold {fold_id} в grouped checkpoint")

            subj_feats = feat_frame[feat_frame["subject_id"] == subject_id]
            subj_feats = subj_feats.merge(
                pred_sub[["sample_index"]].drop_duplicates(),
                on="sample_index",
                how="inner",
            ).sort_values("sample_index")
            if subj_feats.empty:
                continue

            bg_feats = feat_frame[feat_frame["subject_id"] != subject_id][feat_cols]
            bg_feats = _sample_rows(bg_feats, cfg.shap.background_max_samples, rng)

            pipeline = fold_states[fold_id]
            imp = pipeline["imputer"]
            scaler = pipeline["scaler"]
            model = pipeline["model"]

            x_eval_raw = subj_feats[feat_cols].copy()
            x_bg = scaler.transform(imp.transform(bg_feats.to_numpy(dtype=float)))
            x_eval = scaler.transform(imp.transform(x_eval_raw.to_numpy(dtype=float)))

            shap_values, base_values = _compute_shap_values(model, x_bg, x_eval, cfg)
            y_pred = np.asarray(model.predict(x_eval), dtype=float)

            subject_rows.extend(_aggregate_subject_feature_rows(
                model_row=model_row,
                subject_id=str(subject_id),
                feature_names=feat_cols,
                x_eval_raw=x_eval_raw,
                y_pred=y_pred,
                base_values=base_values,
                shap_values=shap_values,
            ))

    subject_df = pd.DataFrame(subject_rows)
    if subject_df.empty:
        global_df = pd.DataFrame()
        modality_df = pd.DataFrame()
    else:
        subject_df = _normalise_share(
            subject_df,
            ["model_id", "subject_id"],
        )
        subject_df["feature_rank_within_subject"] = (
            subject_df.groupby(["model_id", "subject_id"])["mean_abs_shap"]
            .rank(method="dense", ascending=False)
            .astype(int)
        )

        global_df = (
            subject_df.groupby(
                ["model_id", "architecture_id", "family", "target", "feature_set",
                 "with_abs", "phase_split", "feature_name", "modality"],
                as_index=False,
            )
            .agg(
                mean_abs_shap=("mean_abs_shap", "mean"),
                mean_shap=("mean_shap", "mean"),
                mean_feature_value=("mean_feature_value", "mean"),
                mean_prediction=("mean_prediction", "mean"),
                mean_base_value=("mean_base_value", "mean"),
                mean_windows_per_subject=("n_windows", "mean"),
                n_subjects=("subject_id", "nunique"),
            )
        )
        global_df = _normalise_share(global_df, ["model_id"])
        global_df["feature_rank_global"] = (
            global_df.groupby("model_id")["mean_abs_shap"]
            .rank(method="dense", ascending=False)
            .astype(int)
        )

        modality_df = (
            subject_df.groupby(
                ["model_id", "architecture_id", "family", "target", "feature_set",
                 "with_abs", "phase_split", "subject_id", "modality"],
                as_index=False,
            )
            .agg(
                mean_abs_shap=("mean_abs_shap", "sum"),
                mean_shap=("mean_shap", "sum"),
                abs_shap_share=("abs_shap_share", "sum"),
                n_windows=("n_windows", "max"),
            )
        )
        modality_df["modality_rank_within_subject"] = (
            modality_df.groupby(["model_id", "subject_id"])["mean_abs_shap"]
            .rank(method="dense", ascending=False)
            .astype(int)
        )

    cfg.shap_subject_summary_path.parent.mkdir(parents=True, exist_ok=True)
    subject_df.to_parquet(cfg.shap_subject_summary_path, index=False)
    global_df.to_parquet(cfg.shap_global_summary_path, index=False)
    modality_df.to_parquet(cfg.shap_modality_summary_path, index=False)
    return {
        "global": global_df,
        "subject": subject_df,
        "modality": modality_df,
    }


def plot_shap_figures(global_df: pd.DataFrame,
                      modality_df: pd.DataFrame,
                      cfg: AnalysisConfig) -> list[Path]:
    """Строит базовые SHAP-фигуры по finalist-моделям."""
    out: list[Path] = []
    if global_df.empty:
        return out

    out_dir = cfg.figures_dir / "shap"
    out_dir.mkdir(parents=True, exist_ok=True)

    for model_id, g in global_df.groupby("model_id", sort=True):
        top = g.sort_values("mean_abs_shap", ascending=False).head(15)
        fig, ax = plt.subplots(figsize=(9, max(4, len(top) * 0.33)))
        ax.barh(top["feature_name"], top["mean_abs_shap"], color="#2f6db3")
        ax.invert_yaxis()
        ax.set_xlabel("mean(|SHAP|)")
        ax.set_ylabel("Признак")
        title_target = str(top["target"].iloc[0]).upper()
        ax.set_title(f"{model_id}: глобальная важность SHAP ({title_target})")
        fig.tight_layout()
        path = out_dir / f"{model_id}_top_features.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        out.append(path)

    if not modality_df.empty:
        modality_global = (
            modality_df.groupby(["model_id", "target", "modality"], as_index=False)
            .agg(mean_abs_shap=("mean_abs_shap", "mean"))
        )
        for model_id, g in modality_global.groupby("model_id", sort=True):
            g = g.sort_values("mean_abs_shap", ascending=False)
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.bar(g["modality"], g["mean_abs_shap"], color="#4d9c7d")
            ax.set_ylabel("mean(|SHAP|)")
            ax.set_xlabel("Модальность")
            title_target = str(g["target"].iloc[0]).upper()
            ax.set_title(f"{model_id}: SHAP по модальностям ({title_target})")
            fig.tight_layout()
            path = out_dir / f"{model_id}_modalities.png"
            fig.savefig(path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            out.append(path)

    return out
