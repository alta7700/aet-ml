"""v0011_viz — Визуализация предсказаний: траектории, ошибки, MAE per subject.

Использует оптимальные σ из v0011b (σ_p=5, σ_obs=500).

Генерирует:
  results/v0011/viz/trajectories_{target}.png  — все субъекты, 4 набора фичей
  results/v0011/viz/error_vs_dist_{target}.png — box plot ошибки по удалённости
  results/v0011/viz/mae_heatmap_{target}.png   — тепловая карта MAE per subject
  results/v0011/viz/calibration_{target}.png   — predicted vs actual scatter

Запуск:
  uv run python scripts/v0011_viz.py
  uv run python scripts/v0011_viz.py --target lt2
  uv run python scripts/v0011_viz.py --sigma-p 5 --sigma-obs 500
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path
import sys

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from dataset_pipeline.common import DEFAULT_DATASET_DIR
sys.path.insert(0, str(_ROOT / "scripts"))
from v0011_modality_ablation import (
    prepare_data,
    get_feature_cols,
    kalman_smooth,
)

OUT_DIR = _ROOT / "results" / "v0011" / "viz"

# ─── Оптимальные σ из v0011b ─────────────────────────────────────────────────

SIGMA_P_DEFAULT   = 5.0
SIGMA_OBS_DEFAULT = 500.0

TARGET_CFG = {
    "lt2": "target_time_to_lt2_center_sec",
    "lt1": "target_time_to_lt1_pchip_sec",
}

# ─── Лучшие модели из v0011 ───────────────────────────────────────────────────

def _best_factory(feature_set: str, target: str):
    key = (feature_set, target)
    m = {
        ("EMG",          "lt2"): lambda: ElasticNet(alpha=1.0, l1_ratio=0.2, max_iter=5000, random_state=42),
        ("EMG",          "lt1"): lambda: Ridge(alpha=1000),
        ("NIRS",         "lt2"): lambda: SVR(kernel="rbf", C=10, epsilon=0.1),
        ("NIRS",         "lt1"): lambda: SVR(kernel="rbf", C=10, epsilon=1.0),
        ("EMG+NIRS",     "lt2"): lambda: ElasticNet(alpha=1.0, l1_ratio=0.2, max_iter=5000, random_state=42),
        ("EMG+NIRS",     "lt1"): lambda: Ridge(alpha=1000),
        ("EMG+NIRS+HRV", "lt2"): lambda: ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=5000, random_state=42),
        ("EMG+NIRS+HRV", "lt1"): lambda: GradientBoostingRegressor(n_estimators=50, max_depth=2, random_state=42),
    }
    return m[key]

# ─── Цвета наборов ───────────────────────────────────────────────────────────

FSET_COLORS = {
    "EMG":          "#E07B39",
    "NIRS":         "#5B9BD5",
    "EMG+NIRS":     "#70AD47",
    "EMG+NIRS+HRV": "#7030A0",
}
FSET_LABELS = {
    "EMG":          "EMG+Kin",
    "NIRS":         "NIRS",
    "EMG+NIRS":     "EMG+NIRS",
    "EMG+NIRS+HRV": "EMG+NIRS+HRV",
}


# ─── LOSO → per-subject предсказания ─────────────────────────────────────────

def loso_predictions(df: pd.DataFrame, feat_cols: list[str],
                     target_col: str, model_factory,
                     sigma_p: float, sigma_obs: float) -> dict[str, dict]:
    """Возвращает per-subject: t_sec, y_true_min, y_raw_min, y_kalman_min."""
    subjects = sorted(df["subject_id"].unique())
    result: dict[str, dict] = {}

    for test_s in subjects:
        train = df[df["subject_id"] != test_s]
        test  = df[df["subject_id"] == test_s].sort_values("window_start_sec")

        imp = SimpleImputer(strategy="median")
        sc  = StandardScaler()
        mdl = model_factory()

        X_tr = sc.fit_transform(imp.fit_transform(train[feat_cols].values))
        X_te = sc.transform(imp.transform(test[feat_cols].values))
        mdl.fit(X_tr, train[target_col].values)

        y_pred = mdl.predict(X_te)
        y_true = test[target_col].values
        t_sec  = test["window_start_sec"].values

        result[test_s] = {
            "t_sec":       t_sec,
            "y_true_min":  y_true / 60.0,
            "y_raw_min":   y_pred / 60.0,
            "y_kalman_min": kalman_smooth(y_pred, sigma_p, sigma_obs) / 60.0,
        }

    return result


def collect_all_sets(df: pd.DataFrame, session_params: pd.DataFrame,
                     target: str, feature_sets: list[str],
                     sigma_p: float, sigma_obs: float) -> dict[str, dict]:
    """Запускает LOSO для всех feature_sets, возвращает {fset: {subj: data}}."""
    target_col = TARGET_CFG[target]
    df_prep = prepare_data(df, session_params, target)
    if target_col not in df_prep.columns:
        raise RuntimeError(f"Нет {target_col}")
    df_prep = df_prep.dropna(subset=[target_col])

    all_preds: dict[str, dict] = {}
    for fset in feature_sets:
        feat_cols = get_feature_cols(df_prep, fset)
        if not feat_cols:
            continue
        factory = _best_factory(fset, target)
        print(f"  LOSO [{fset}]...", end=" ", flush=True)
        preds = loso_predictions(df_prep, feat_cols, target_col,
                                 factory, sigma_p, sigma_obs)
        mae_all = np.mean([
            mean_absolute_error(v["y_true_min"], v["y_kalman_min"])
            for v in preds.values()
        ])
        print(f"MAE={mae_all:.3f} мин")
        all_preds[fset] = preds

    return all_preds, df_prep


# ─── 1. Траектории ────────────────────────────────────────────────────────────

def plot_trajectories(all_preds: dict, subjects: list[str],
                      target: str, out_dir: Path) -> None:
    """Сетка субъектов × наборов: траектория предсказания vs истина."""
    n_subj = len(subjects)
    n_cols = 4
    n_rows = int(np.ceil(n_subj / n_cols))

    fsets_ordered = [f for f in ["EMG", "NIRS", "EMG+NIRS", "EMG+NIRS+HRV"]
                     if f in all_preds]

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 3.5 * n_rows),
                             squeeze=False)
    fig.suptitle(f"Траектории предсказания времени до порога — {target.upper()}",
                 fontsize=14, y=1.01)

    for idx, subj in enumerate(subjects):
        ax = axes[idx // n_cols][idx % n_cols]

        # Истинная траектория (одинаковая для всех наборов)
        ref_fset = fsets_ordered[0]
        if subj not in all_preds[ref_fset]:
            ax.set_visible(False)
            continue

        t_min = all_preds[ref_fset][subj]["t_sec"] / 60.0
        y_true = all_preds[ref_fset][subj]["y_true_min"]

        ax.plot(t_min, y_true, color="black", linewidth=2.0,
                label="Истина", zorder=10)

        for fset in fsets_ordered:
            if subj not in all_preds[fset]:
                continue
            color = FSET_COLORS[fset]
            data  = all_preds[fset][subj]

            # Сырые предсказания — тонкие, полупрозрачные
            ax.plot(data["t_sec"] / 60.0, data["y_raw_min"],
                    color=color, alpha=0.25, linewidth=1.0, linestyle="--")
            # Kalman — жирные
            ax.plot(data["t_sec"] / 60.0, data["y_kalman_min"],
                    color=color, alpha=0.9, linewidth=1.8,
                    label=FSET_LABELS[fset])

        ax.set_title(subj, fontsize=9, pad=3)
        ax.set_xlabel("Время теста (мин)", fontsize=7)
        ax.set_ylabel("Время до порога (мин)", fontsize=7)
        ax.tick_params(labelsize=7)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(bottom=0)

    # Скрыть пустые ячейки
    for idx in range(len(subjects), n_rows * n_cols):
        axes[idx // n_cols][idx % n_cols].set_visible(False)

    # Легенда
    legend_elements = [
        plt.Line2D([0], [0], color="black", linewidth=2.0, label="Истина"),
    ] + [
        plt.Line2D([0], [0], color=FSET_COLORS[f], linewidth=1.8,
                   label=FSET_LABELS[f])
        for f in fsets_ordered
    ] + [
        plt.Line2D([0], [0], color="gray", linewidth=1.0, linestyle="--",
                   alpha=0.5, label="Сырое (до Калмана)")
    ]
    fig.legend(handles=legend_elements, loc="lower center", ncol=len(legend_elements),
               fontsize=9, bbox_to_anchor=(0.5, -0.02))

    plt.tight_layout()
    path = out_dir / f"trajectories_{target}.png"
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {path.name}")


# ─── 2. Ошибка vs удалённость до порога ──────────────────────────────────────

def plot_error_vs_dist(all_preds: dict, target: str, out_dir: Path) -> None:
    """Box plot абсолютной ошибки по бакетам истинного времени до порога."""
    buckets = [(0, 5), (5, 10), (10, 20), (20, 999)]
    bucket_labels = ["0–5 мин", "5–10 мин", "10–20 мин", ">20 мин"]

    fsets_ordered = [f for f in ["EMG", "NIRS", "EMG+NIRS", "EMG+NIRS+HRV"]
                     if f in all_preds]
    n_fsets = len(fsets_ordered)

    fig, axes = plt.subplots(1, n_fsets, figsize=(4.5 * n_fsets, 5),
                             sharey=False, squeeze=False)
    fig.suptitle(f"|Ошибка| (мин) vs удалённость до порога — {target.upper()}",
                 fontsize=13)

    for fi, fset in enumerate(fsets_ordered):
        ax = axes[0][fi]
        data_by_bucket: list[list[float]] = [[] for _ in buckets]

        for subj, subj_data in all_preds[fset].items():
            y_true   = subj_data["y_true_min"]
            y_kalman = subj_data["y_kalman_min"]
            abs_err  = np.abs(y_true - y_kalman)

            for bi, (lo, hi) in enumerate(buckets):
                mask = (y_true >= lo) & (y_true < hi)
                data_by_bucket[bi].extend(abs_err[mask].tolist())

        # Убрать пустые бакеты
        non_empty = [(bucket_labels[bi], data_by_bucket[bi])
                     for bi in range(len(buckets)) if data_by_bucket[bi]]

        bp = ax.boxplot(
            [d for _, d in non_empty],
            labels=[l for l, _ in non_empty],
            patch_artist=True,
            widths=0.55,
            medianprops=dict(color="black", linewidth=2),
            flierprops=dict(marker="o", markersize=3, alpha=0.4),
        )
        color = FSET_COLORS[fset]
        for patch in bp["boxes"]:
            patch.set_facecolor(color)
            patch.set_alpha(0.6)

        # Аннотация медианы и числа наблюдений
        # Фиксируем y_max = 95-й перцентиль всех данных в наборе
        all_vals = [v for _, d in non_empty for v in d]
        y_max = np.percentile(all_vals, 95) * 1.25 if all_vals else 12
        ax.set_ylim(0, y_max)
        for i, (_, d) in enumerate(non_empty):
            med = np.median(d)
            n   = len(d)
            ann_y = min(med + 0.15, y_max * 0.90)
            ax.text(i + 1, ann_y, f"{med:.2f}", ha="center",
                    va="bottom", fontsize=8, fontweight="bold",
                    clip_on=True)
            ax.text(i + 1, y_max * 0.03, f"n={n}",
                    ha="center", va="bottom", fontsize=7, color="gray")

        ax.set_title(FSET_LABELS[fset], fontsize=11,
                     color=color, fontweight="bold")
        ax.set_xlabel("Истинное время до порога", fontsize=9)
        if fi == 0:
            ax.set_ylabel("|Ошибка| (мин)", fontsize=9)
        ax.tick_params(labelsize=8, axis="x", rotation=15)
        ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    path = out_dir / f"error_vs_dist_{target}.png"
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {path.name}")


# ─── 3. MAE per subject тепловая карта ───────────────────────────────────────

def plot_mae_heatmap(all_preds: dict, subjects: list[str],
                     target: str, out_dir: Path) -> None:
    """Тепловая карта MAE: строки = feature_set, столбцы = субъекты."""
    fsets_ordered = [f for f in ["EMG", "NIRS", "EMG+NIRS", "EMG+NIRS+HRV"]
                     if f in all_preds]

    mae_matrix = np.full((len(fsets_ordered), len(subjects)), np.nan)
    for fi, fset in enumerate(fsets_ordered):
        for si, subj in enumerate(subjects):
            if subj not in all_preds[fset]:
                continue
            d = all_preds[fset][subj]
            mae_matrix[fi, si] = mean_absolute_error(
                d["y_true_min"], d["y_kalman_min"])

    fig, ax = plt.subplots(figsize=(max(10, len(subjects) * 0.85), 3.5))

    vmin = np.nanmin(mae_matrix)
    vmax = np.nanmax(mae_matrix)
    im = ax.imshow(mae_matrix, aspect="auto", cmap="RdYlGn_r",
                   vmin=vmin, vmax=vmax, interpolation="nearest")
    plt.colorbar(im, ax=ax, label="MAE (мин)")

    ax.set_xticks(range(len(subjects)))
    ax.set_xticklabels(subjects, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(len(fsets_ordered)))
    ax.set_yticklabels([FSET_LABELS[f] for f in fsets_ordered], fontsize=10)
    ax.set_title(f"MAE per subject (мин) — {target.upper()}", fontsize=12)

    # Значения в ячейках
    for fi in range(len(fsets_ordered)):
        for si in range(len(subjects)):
            v = mae_matrix[fi, si]
            if np.isfinite(v):
                ax.text(si, fi, f"{v:.2f}", ha="center", va="center",
                        fontsize=8, color="black")

    # Средние по строкам (правее)
    for fi, fset in enumerate(fsets_ordered):
        row = mae_matrix[fi]
        row_mean = np.nanmean(row)
        ax.text(len(subjects) + 0.1, fi, f"μ={row_mean:.2f}",
                ha="left", va="center", fontsize=9, color="navy",
                fontweight="bold")

    ax.set_xlim(-0.5, len(subjects) + 0.9)

    plt.tight_layout()
    path = out_dir / f"mae_heatmap_{target}.png"
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {path.name}")


# ─── 4. Калибровка: predicted vs actual ──────────────────────────────────────

def plot_calibration(all_preds: dict, target: str, out_dir: Path) -> None:
    """Scatter: predicted (Kalman) vs actual, per feature_set."""
    fsets_ordered = [f for f in ["EMG", "NIRS", "EMG+NIRS", "EMG+NIRS+HRV"]
                     if f in all_preds]
    n_fsets = len(fsets_ordered)

    fig, axes = plt.subplots(1, n_fsets, figsize=(4.5 * n_fsets, 4.5),
                             squeeze=False)
    fig.suptitle(f"Калибровка: предсказание vs истина — {target.upper()}",
                 fontsize=13)

    for fi, fset in enumerate(fsets_ordered):
        ax = axes[0][fi]
        color = FSET_COLORS[fset]

        all_true, all_pred = [], []
        for subj, d in all_preds[fset].items():
            all_true.extend(d["y_true_min"].tolist())
            all_pred.extend(d["y_kalman_min"].tolist())

        all_true = np.array(all_true)
        all_pred = np.array(all_pred)

        # Scatter по субъектам (разные маркеры не нужны — просто прозрачные точки)
        ax.scatter(all_true, all_pred, color=color, alpha=0.25,
                   s=8, edgecolors="none")

        # Линия идеального предсказания
        lo = min(all_true.min(), all_pred.min())
        hi = max(all_true.max(), all_pred.max())
        ax.plot([lo, hi], [lo, hi], "k--", linewidth=1.5, label="Идеал")

        # Линия регрессии
        coef = np.polyfit(all_true, all_pred, 1)
        x_fit = np.linspace(lo, hi, 100)
        ax.plot(x_fit, np.polyval(coef, x_fit),
                color=color, linewidth=2.0, alpha=0.9,
                label=f"Тренд (k={coef[0]:.2f})")

        mae = mean_absolute_error(all_true, all_pred)
        corr = np.corrcoef(all_true, all_pred)[0, 1]
        ax.set_title(f"{FSET_LABELS[fset]}\nMAE={mae:.2f} мин  r={corr:.3f}",
                     fontsize=10, color=color, fontweight="bold")
        ax.set_xlabel("Истинное время до порога (мин)", fontsize=9)
        if fi == 0:
            ax.set_ylabel("Предсказанное время (мин)", fontsize=9)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_aspect("equal", adjustable="box")

    plt.tight_layout()
    path = out_dir / f"calibration_{target}.png"
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {path.name}")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="v0011 визуализации")
    parser.add_argument("--target",    choices=["lt1", "lt2", "both"], default="both")
    parser.add_argument("--sigma-p",   type=float, default=SIGMA_P_DEFAULT)
    parser.add_argument("--sigma-obs", type=float, default=SIGMA_OBS_DEFAULT)
    parser.add_argument("--dataset",
                        default=DEFAULT_DATASET_DIR / "merged_features_ml.parquet")
    parser.add_argument("--feature-set", nargs="+",
                        choices=["EMG", "NIRS", "EMG+NIRS", "EMG+NIRS+HRV"],
                        default=["EMG", "NIRS", "EMG+NIRS", "EMG+NIRS+HRV"])
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Загружаю {args.dataset}")
    print(f"σ_p={args.sigma_p}, σ_obs={args.sigma_obs}")
    df_raw = pd.read_parquet(args.dataset)
    sp_path = DEFAULT_DATASET_DIR / "session_params.parquet"
    session_params = pd.read_parquet(sp_path) if sp_path.exists() else pd.DataFrame()

    targets = ["lt2", "lt1"] if args.target == "both" else [args.target]

    for tgt in targets:
        print(f"\n{'═'*60}")
        print(f"ТАРГЕТ: {tgt.upper()}")
        print(f"{'═'*60}")

        all_preds, df_prep = collect_all_sets(
            df_raw, session_params, tgt, args.feature_set,
            args.sigma_p, args.sigma_obs)

        subjects = sorted(df_prep["subject_id"].unique())
        print(f"Субъектов: {len(subjects)}")
        print("Генерирую графики...")

        plot_trajectories(all_preds, subjects, tgt, OUT_DIR)
        plot_error_vs_dist(all_preds, tgt, OUT_DIR)
        plot_mae_heatmap(all_preds, subjects, tgt, OUT_DIR)
        plot_calibration(all_preds, tgt, OUT_DIR)

    print(f"\n✓ Все графики в {OUT_DIR}/")


if __name__ == "__main__":
    main()
