"""Шаг 02b: анализ ошибки по времени до порога (time-resolved).

Берёт топ-3 модели на target, восстанавливает per-window ошибки из npy,
биннингует по `time_to_threshold` (секунды до LT) и `elapsed_sec`
(от начала рабочей части теста). Строит кривые медианной ошибки.

Артефакты — в results/final/eda/time_resolved/.

Запуск:
    PYTHONPATH=. uv run python scripts/final_time_resolved.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from v0011_modality_ablation import prepare_data  # noqa: E402

DATASET_DIR = ROOT / "dataset"
RESULTS = ROOT / "results"
EDA = RESULTS / "final" / "eda" / "time_resolved"
PLOTS = EDA / "plots"

TARGET_COL = {
    "lt1": "target_time_to_lt1_pchip_sec",
    "lt2": "target_time_to_lt2_center_sec",
}

# Топ-модели: (version, variant, fset_label, fset_tag, target, short_id)
TOP_MODELS = [
    ("v0011", "with_abs", "HRV",          "HRV",          "lt2", "v0011_HRV"),
    ("v0011", "with_abs", "EMG+NIRS+HRV", "EMG_NIRS_HRV", "lt2", "v0011_ENH"),
    ("v0107", "with_abs", "EMG+NIRS+HRV", "EMG_NIRS_HRV", "lt2", "v0107_ENH"),
    ("v0011", "with_abs", "EMG+NIRS",     "EMG_NIRS",     "lt1", "v0011_EN_lt1"),
    ("v0011", "with_abs", "EMG+NIRS+HRV", "EMG_NIRS_HRV", "lt1", "v0011_ENH_lt1"),
    ("v0107", "noabs",    "EMG+NIRS+HRV", "EMG_NIRS_HRV", "lt1", "v0107_ENH_lt1"),
]

# Бины по времени до порога (секунды): 0–30, 30–60, 60–120, 120–240, 240–480, 480+
BIN_EDGES = [0, 30, 60, 120, 240, 480, 99999]
BIN_LABELS = ["0-30s", "30-60s", "60-120s", "120-240s", "240-480s", ">480s"]


def build_df_target(df_raw: pd.DataFrame, sp: pd.DataFrame, target: str) -> pd.DataFrame:
    df = prepare_data(df_raw, sp, target)
    tcol = TARGET_COL[target]
    if tcol not in df.columns:
        return pd.DataFrame()
    df = df.dropna(subset=[tcol]).copy()
    return df[["subject_id", "window_id", "window_start_sec", tcol]]


def load_old_per_subject(version: str) -> pd.DataFrame:
    """Список субъектов для каждой комбинации (variant, fset, target)."""
    path = RESULTS / version / "per_subject.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    if version == "v0011":
        df["variant"] = "with_abs"
        return df[["variant", "feature_set", "target", "subject_id"]]
    if version == "v0107":
        return df[["variant", "feature_set", "target", "subject_id"]]
    return df[["variant", "feature_set", "target", "subject_id"]]


def load_predictions(version: str, variant: str, fset_tag: str,
                     target: str, df_target: pd.DataFrame,
                     subjects: list[str]) -> pd.DataFrame:
    """Возвращает df с per-window: subject_id, time_to_threshold, abs_err."""
    vdir = RESULTS / version / ("noabs" if variant == "noabs" else "")
    if variant == "with_abs":
        vdir = RESULTS / version
    yp_path = vdir / f"ypred_{target}_{fset_tag}.npy"
    yt_path = vdir / f"ytrue_{target}_{fset_tag}.npy"
    if not (yp_path.exists() and yt_path.exists()):
        return pd.DataFrame()

    yp = np.load(yp_path)
    yt = np.load(yt_path)

    df_subset = (df_target[df_target["subject_id"].isin(subjects)]
                 .sort_values(["subject_id", "window_start_sec"])
                 .reset_index(drop=True))
    if len(df_subset) != len(yp):
        # Перебираем subset, как в final_per_subject_from_npy
        from scripts.final_per_subject_from_npy import determine_subjects
        subj_list = determine_subjects(df_target, len(yp), subjects)
        if subj_list is None:
            return pd.DataFrame()
        df_subset = (df_target[df_target["subject_id"].isin(subj_list)]
                     .sort_values(["subject_id", "window_start_sec"])
                     .reset_index(drop=True))
        if len(df_subset) != len(yp):
            return pd.DataFrame()

    tcol = TARGET_COL[target]
    df_subset["y_pred"] = yp
    df_subset["y_true"] = yt
    df_subset["abs_err_sec"] = np.abs(yp - yt)
    df_subset["time_to_threshold_sec"] = df_subset[tcol]
    df_subset["elapsed_sec"] = df_subset["window_start_sec"]
    return df_subset[["subject_id", "elapsed_sec", "time_to_threshold_sec",
                      "abs_err_sec", "y_pred", "y_true"]]


def bin_summary(df: pd.DataFrame, x_col: str,
                bin_edges: list[float], bin_labels: list[str]) -> pd.DataFrame:
    """Биннинг df[x_col] и агрегация ошибки."""
    df = df.copy()
    df["bin"] = pd.cut(df[x_col], bins=bin_edges, labels=bin_labels,
                       include_lowest=True, right=False)
    g = df.groupby("bin", observed=True).agg(
        n=("abs_err_sec", "count"),
        median_err_sec=("abs_err_sec", "median"),
        q25_err_sec=("abs_err_sec", lambda s: float(np.quantile(s, 0.25))),
        q75_err_sec=("abs_err_sec", lambda s: float(np.quantile(s, 0.75))),
    ).reset_index()
    g["median_err_min"] = (g["median_err_sec"] / 60.0).round(3)
    return g


def plot_error_vs_x(per_model: dict[str, pd.DataFrame], x_col: str,
                    xlabel: str, target: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for mid, df in per_model.items():
        s = bin_summary(df, x_col, BIN_EDGES, BIN_LABELS)
        x_pos = np.arange(len(s))
        ax.plot(x_pos, s["median_err_sec"] / 60.0, "o-", label=mid, linewidth=2)
        ax.fill_between(x_pos, s["q25_err_sec"] / 60.0, s["q75_err_sec"] / 60.0,
                        alpha=0.15)
    ax.set_xticks(np.arange(len(BIN_LABELS)))
    ax.set_xticklabels(BIN_LABELS)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("|ошибка|, мин (median, IQR)")
    ax.set_title(f"{target.upper()} — ошибка vs {xlabel}")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_per_subject_curves(df: pd.DataFrame, target: str, mid: str,
                             path: Path) -> None:
    """Кривая ошибки vs time_to_threshold по каждому субъекту + жирная медиана."""
    fig, ax = plt.subplots(figsize=(10, 5.5))
    df = df.copy()
    df["bin"] = pd.cut(df["time_to_threshold_sec"], bins=BIN_EDGES,
                       labels=BIN_LABELS, include_lowest=True, right=False)
    for sid, g in df.groupby("subject_id"):
        med = g.groupby("bin", observed=True)["abs_err_sec"].median() / 60.0
        ax.plot(np.arange(len(BIN_LABELS)),
                med.reindex(BIN_LABELS).values,
                color="gray", alpha=0.4, linewidth=1)
    med_all = df.groupby("bin", observed=True)["abs_err_sec"].median() / 60.0
    ax.plot(np.arange(len(BIN_LABELS)),
            med_all.reindex(BIN_LABELS).values,
            "ro-", linewidth=2.5, label="медиана по выборке")
    ax.set_xticks(np.arange(len(BIN_LABELS)))
    ax.set_xticklabels(BIN_LABELS)
    ax.set_xlabel("Время до порога")
    ax.set_ylabel("|ошибка|, мин")
    ax.set_title(f"{mid} ({target.upper()}) — per-subject curves")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main() -> None:
    EDA.mkdir(parents=True, exist_ok=True)
    PLOTS.mkdir(parents=True, exist_ok=True)

    print("Подготовка датасета…")
    df_raw = pd.read_parquet(DATASET_DIR / "merged_features_ml.parquet")
    sp = pd.read_parquet(DATASET_DIR / "session_params.parquet")
    df_by_target = {t: build_df_target(df_raw, sp, t) for t in ("lt1", "lt2")}

    all_summaries: list[pd.DataFrame] = []
    per_model_by_target = {"lt1": {}, "lt2": {}}

    for version, variant, fset_label, fset_tag, target, mid in TOP_MODELS:
        print(f"\n[{mid}] {version}/{variant}/{target}/{fset_label}")
        old = load_old_per_subject(version)
        if old.empty:
            subjects = sorted(df_by_target[target]["subject_id"].unique())
        else:
            m = ((old["target"] == target)
                 & (old["variant"] == variant)
                 & (old["feature_set"] == fset_label))
            subjects = sorted(old.loc[m, "subject_id"].unique().tolist()) \
                       or sorted(df_by_target[target]["subject_id"].unique())

        preds = load_predictions(version, variant, fset_tag, target,
                                  df_by_target[target], subjects)
        if preds.empty:
            print("  (нет предсказаний)")
            continue
        print(f"  окон: {len(preds)}, субъектов: {preds['subject_id'].nunique()}")
        per_model_by_target[target][mid] = preds

        # Биннинг по времени до порога
        s_tt = bin_summary(preds, "time_to_threshold_sec", BIN_EDGES, BIN_LABELS)
        s_tt["model_id"] = mid
        s_tt["target"] = target
        s_tt["x"] = "time_to_threshold"
        all_summaries.append(s_tt)

        # Биннинг по elapsed_sec (контроль)
        s_el = bin_summary(preds, "elapsed_sec", BIN_EDGES, BIN_LABELS)
        s_el["model_id"] = mid
        s_el["target"] = target
        s_el["x"] = "elapsed_sec"
        all_summaries.append(s_el)

        # Per-subject curves
        plot_per_subject_curves(
            preds, target, mid,
            PLOTS / f"per_subject_curves_{mid}.png",
        )

    if all_summaries:
        summary_df = pd.concat(all_summaries, ignore_index=True)
        summary_path = EDA / "10_error_by_time_bins.csv"
        summary_df.to_csv(summary_path, index=False)
        print(f"\n→ {summary_path.name}")

    # Сводные графики
    for target in ("lt1", "lt2"):
        pm = per_model_by_target[target]
        if pm:
            plot_error_vs_x(pm, "time_to_threshold_sec",
                             "время до порога",
                             target,
                             PLOTS / f"error_vs_time_to_threshold_{target}.png")
            plot_error_vs_x(pm, "elapsed_sec",
                             "elapsed от начала теста",
                             target,
                             PLOTS / f"error_vs_elapsed_{target}.png")

    print("Готово.")


if __name__ == "__main__":
    main()
