"""eval_utils.py — Расширенные метрики и визуализации для оценки моделей.

Не версионируется (не обучает модели), отслеживается через git.

Метрики:
  Acc@δ          — доля окон, где |ŷ - y| ≤ δ (accuracy within tolerance)
  Acc@δ(t_norm)  — то же, разбитое по нормализованному времени теста
  TDE            — Threshold Detection Error: ошибка в определении момента порога

Использование:
  from eval_utils import compute_all_metrics, plot_acc_by_time, plot_tde, plot_summary
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ─── Acc@δ ────────────────────────────────────────────────────────────────────

def acc_within_tol(y_true: np.ndarray, y_pred: np.ndarray, delta_sec: float) -> float:
    """Доля предсказаний с |ŷ - y| ≤ δ (в секундах)."""
    return float(np.mean(np.abs(y_true - y_pred) <= delta_sec))


# ─── Threshold Detection Error ────────────────────────────────────────────────

def threshold_detection_error(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    elapsed: np.ndarray,
    subjects: np.ndarray,
) -> dict[str, float]:
    """Ошибка детектирования порога для каждого участника (в секундах).

    Подход: каждое окно даёт оценку абсолютного момента порога:
      T_est(i) = elapsed(i) + y(i)   (elapsed + time_remaining = abs. threshold time)

    Усредняем по всем окнам участника:
      T̂_pred = mean(elapsed + y_pred)  — модельная оценка
      T_true  = mean(elapsed + y_true)  — истинная (почти константа внутри сессии)

    TDE = T̂_pred - T_true  (сек), положительный → модель «опаздывает».
    """
    result: dict[str, float] = {}
    for s in sorted(np.unique(subjects)):
        m = subjects == s
        t_pred = float(np.mean(elapsed[m] + y_pred[m]))
        t_true = float(np.mean(elapsed[m] + y_true[m]))
        result[s] = t_pred - t_true
    return result


# ─── Acc@δ по нормализованному времени теста ─────────────────────────────────

def acc_by_normalized_time(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    elapsed: np.ndarray,
    subjects: np.ndarray,
    deltas_sec: list[float],
    n_bins: int = 10,
) -> dict:
    """Acc@δ разбитый по нормализованному времени теста.

    t_norm = elapsed / max_elapsed_per_subject ∈ [0, 1].
    Делит тест на n_bins равных частей, для каждой считает Acc@δ.

    Возвращает:
      bin_centers: np.ndarray (n_bins,)
      acc: dict[delta → np.ndarray (n_bins,)]  — NaN если бин пустой
      counts: np.ndarray (n_bins,)
    """
    # Нормализованное время для каждого окна
    t_norm = np.zeros(len(elapsed))
    for s in np.unique(subjects):
        m = subjects == s
        total = elapsed[m].max()
        t_norm[m] = elapsed[m] / total if total > 0 else 0.0

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_centers = (bins[:-1] + bins[1:]) / 2.0
    counts = np.zeros(n_bins, dtype=int)
    acc: dict[float, np.ndarray] = {d: np.full(n_bins, np.nan) for d in deltas_sec}

    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        # Последний бин включает правую границу
        mask = (t_norm >= lo) & (t_norm < hi) if i < n_bins - 1 else (t_norm >= lo) & (t_norm <= hi)
        counts[i] = int(mask.sum())
        if counts[i] > 0:
            for d in deltas_sec:
                acc[d][i] = acc_within_tol(y_true[mask], y_pred[mask], d)

    return {"bin_centers": bin_centers, "acc": acc, "counts": counts}


# ─── Сводные метрики ──────────────────────────────────────────────────────────

DELTAS_SEC = [30.0, 60.0, 120.0, 180.0]   # 30с, 1мин, 2мин, 3мин
DELTA_LABELS = ["Acc@30s", "Acc@1min", "Acc@2min", "Acc@3min"]


NEAR_THRESHOLD_SEC = 120.0  # ±2 мин от момента порога


def compute_split_mae(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    near_sec: float = NEAR_THRESHOLD_SEC,
) -> dict:
    """MAE разбитый на три зоны по знаку и близости истинного τ.

    mae_pre  — окна до порога  (τ_true > 0): точность прогноза будущего
    mae_post — окна после порога (τ_true ≤ 0): скорость «признания» наступившего порога
    mae_near — окна вблизи порога (|τ_true| ≤ near_sec): качество в зоне перехода
    """
    def _mae(mask: np.ndarray) -> float:
        return float(np.mean(np.abs(y_true[mask] - y_pred[mask]))) / 60.0 if mask.sum() > 0 else float("nan")

    mask_pre  = y_true > 0
    mask_post = y_true <= 0
    mask_near = np.abs(y_true) <= near_sec

    return {
        "mae_pre":  _mae(mask_pre),
        "mae_post": _mae(mask_post),
        "mae_near": _mae(mask_near),
        "n_pre":    int(mask_pre.sum()),
        "n_post":   int(mask_post.sum()),
        "n_near":   int(mask_near.sum()),
    }


def compute_all_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    elapsed: np.ndarray,
    subjects: np.ndarray,
    n_bins: int = 10,
) -> dict:
    """Вычисляет все расширенные метрики для одного LOSO-прогона."""
    tde = threshold_detection_error(y_true, y_pred, elapsed, subjects)
    tde_vals = np.array(list(tde.values()))

    acc_global = {d: acc_within_tol(y_true, y_pred, d) for d in DELTAS_SEC}
    time_bins = acc_by_normalized_time(y_true, y_pred, elapsed, subjects, DELTAS_SEC, n_bins)

    mae_min = float(np.mean(np.abs(y_true - y_pred))) / 60.0
    split = compute_split_mae(y_true, y_pred)

    return {
        "mae_min": mae_min,
        "mae_pre": split["mae_pre"],
        "mae_post": split["mae_post"],
        "mae_near": split["mae_near"],
        "n_pre": split["n_pre"],
        "n_post": split["n_post"],
        "n_near": split["n_near"],
        "acc_global": acc_global,
        "tde": tde,
        "tde_mean_abs_min": float(np.mean(np.abs(tde_vals))) / 60.0,
        "tde_median_abs_min": float(np.median(np.abs(tde_vals))) / 60.0,
        "time_bins": time_bins,
    }


# ─── Визуализации ─────────────────────────────────────────────────────────────

_DELTA_COLORS = {30.0: "#d62728", 60.0: "#ff7f0e", 120.0: "#2ca02c", 180.0: "#1f77b4"}
_DELTA_LABELS = {30.0: "Acc@30s", 60.0: "Acc@1min", 120.0: "Acc@2min", 180.0: "Acc@3min"}


def plot_acc_by_time(
    metrics: dict,
    title: str,
    out_path: Path,
    deltas: list[float] | None = None,
) -> None:
    """Acc@δ в зависимости от нормализованного времени теста.

    x-ось: t_norm = elapsed / total_test_duration ∈ [0, 1]
    y-ось: Acc@δ = доля окон в бине, где |ŷ - y| ≤ δ

    Линия тренда: полином 2-й степени через бины.
    Доверительный интервал: 95% CI по биномиальному нормальному приближению:
      CI = p̂ ± 1.96 × √(p̂(1−p̂)/n)
    """
    if deltas is None:
        deltas = DELTAS_SEC

    time_bins = metrics["time_bins"]
    centers = time_bins["bin_centers"]
    acc_dict = time_bins["acc"]
    counts = time_bins["counts"]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8),
                                   gridspec_kw={"height_ratios": [3, 1]})
    fig.suptitle(title, fontsize=12, fontweight="bold")

    for d in deltas:
        vals = acc_dict[d]
        valid = ~np.isnan(vals) & (counts > 0)
        if valid.sum() < 2:
            continue

        x_v = centers[valid]
        y_v = vals[valid]
        n_v = counts[valid].astype(float)

        # 95% CI: биномиальное нормальное приближение
        se = np.sqrt(np.maximum(y_v * (1 - y_v), 0) / np.maximum(n_v, 1))
        ci_lo = np.clip(y_v - 1.96 * se, 0, 1)
        ci_hi = np.clip(y_v + 1.96 * se, 0, 1)

        # Линия тренда: полином 2-й степени
        x_dense = np.linspace(x_v[0], x_v[-1], 300)
        try:
            deg = min(2, len(x_v) - 1)
            coeffs = np.polyfit(x_v, y_v, deg=deg)
            y_trend = np.clip(np.polyval(coeffs, x_dense), 0, 1)
        except Exception:
            y_trend = np.interp(x_dense, x_v, y_v)

        # CI-полоса: интерполируем на плотную сетку
        ci_lo_dense = np.clip(np.interp(x_dense, x_v, ci_lo), 0, 1)
        ci_hi_dense = np.clip(np.interp(x_dense, x_v, ci_hi), 0, 1)

        color = _DELTA_COLORS[d]
        ax1.fill_between(x_dense, ci_lo_dense, ci_hi_dense, color=color, alpha=0.15)
        ax1.plot(x_dense, y_trend, color=color, lw=2.5, label=_DELTA_LABELS[d])
        ax1.scatter(x_v, y_v, color=color, s=35, zorder=5, edgecolors="white", lw=0.5)

    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1.05)
    ax1.set_ylabel("Accuracy within tolerance", fontsize=11)
    ax1.legend(fontsize=10, loc="upper left")
    ax1.grid(alpha=0.3)
    ax1.set_xticks(centers)
    ax1.set_xticklabels([f"{c:.1f}" for c in centers], fontsize=8)
    ax1.axvline(1.0, color="red", lw=1, ls=":", alpha=0.6)

    # Полоска с количеством окон в бине
    ax2.bar(centers, counts, width=0.08, color="#aec7e8", alpha=0.85, edgecolor="gray", lw=0.5)
    ax2.set_ylabel("Окна (шт.)", fontsize=9)
    ax2.set_xlabel("Нормализованное время теста  (0 = начало теста, 1 = конец)")
    ax2.set_xlim(0, 1)
    ax2.set_xticks(centers)
    ax2.set_xticklabels([f"{c:.1f}" for c in centers], fontsize=8)
    ax2.grid(alpha=0.2, axis="y")

    acc_text = "  ".join(f"{_DELTA_LABELS[d]}={metrics['acc_global'][d]:.1%}" for d in deltas)
    fig.text(0.5, 0.925, acc_text, ha="center", fontsize=9, color="dimgray")

    plt.tight_layout(rect=[0, 0, 1, 0.92])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  → {out_path.relative_to(out_path.parents[2])}")


def plot_tde(
    metrics: dict,
    title: str,
    out_path: Path,
) -> None:
    """Гистограмма Threshold Detection Error по участникам.

    TDE > 0 → модель «опаздывает» (предсказывает порог позже истинного).
    TDE < 0 → «спешит» (предсказывает раньше).
    """
    tde = metrics["tde"]
    subjects = sorted(tde.keys())
    vals_min = np.array([tde[s] / 60.0 for s in subjects])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(title, fontsize=12, fontweight="bold")

    # Bar по участникам
    colors = ["#d62728" if v > 0 else "#2ca02c" for v in vals_min]
    bars = ax1.barh(subjects, vals_min, color=colors, alpha=0.8)
    ax1.axvline(0, color="black", lw=1.2)
    for bar, v in zip(bars, vals_min):
        ha = "left" if v >= 0 else "right"
        offset = 0.05 if v >= 0 else -0.05
        ax1.text(v + offset, bar.get_y() + bar.get_height() / 2,
                 f"{v:+.2f}", va="center", fontsize=8, ha=ha)
    ax1.set_xlabel("TDE, мин (+ = модель опаздывает)")
    ax1.set_title("TDE по участникам")
    ax1.grid(alpha=0.3, axis="x")
    mean_tde = float(np.mean(vals_min))
    ax1.axvline(mean_tde, color="navy", lw=1.5, ls="--", label=f"Среднее {mean_tde:+.2f} мин")
    ax1.legend(fontsize=9)

    # Гистограмма
    ax2.hist(vals_min, bins=8, color="#7f7f7f", alpha=0.7, edgecolor="white")
    ax2.axvline(0, color="black", lw=1.5)
    ax2.axvline(float(np.mean(vals_min)), color="red", lw=1.5, ls="--",
                label=f"μ={np.mean(vals_min):+.2f} мин")
    ax2.axvline(float(np.median(vals_min)), color="orange", lw=1.5, ls=":",
                label=f"median={np.median(vals_min):+.2f} мин")
    ax2.set_xlabel("TDE, мин")
    ax2.set_ylabel("Участников")
    ax2.set_title(f"Распределение TDE\n|TDE| mean={metrics['tde_mean_abs_min']:.2f} мин, "
                  f"median={metrics['tde_median_abs_min']:.2f} мин")
    ax2.legend(fontsize=9)
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  → {out_path.relative_to(out_path.parents[2])}")


def plot_summary_comparison(
    all_metrics: dict[str, dict],
    out_path: Path,
    title: str = "Сравнение версий",
) -> None:
    """Сравнительный график Acc@δ и MAE по версиям."""
    versions = list(all_metrics.keys())
    n = len(versions)
    x = np.arange(n)
    width = 0.18

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(title, fontsize=13, fontweight="bold")

    # Acc@δ grouped bars
    ax = axes[0]
    for j, (d, label) in enumerate(zip(DELTAS_SEC, DELTA_LABELS)):
        vals = [all_metrics[v]["acc_global"][d] for v in versions]
        ax.bar(x + j * width, vals, width, label=label,
               color=_DELTA_COLORS[d], alpha=0.8)
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(versions, rotation=20, ha="right", fontsize=8)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Accuracy within tolerance")
    ax.set_title("Acc@δ по версиям")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3, axis="y")
    for i, v in enumerate(versions):
        for j, d in enumerate(DELTAS_SEC):
            val = all_metrics[v]["acc_global"][d]
            ax.text(i + j * width, val + 0.01, f"{val:.0%}",
                    ha="center", va="bottom", fontsize=6, rotation=90)

    # MAE + |TDE| bars
    ax2 = axes[1]
    mae_vals = [all_metrics[v]["mae_min"] for v in versions]
    tde_vals = [all_metrics[v]["tde_mean_abs_min"] for v in versions]
    ax2.bar(x - 0.2, mae_vals, 0.35, label="MAE (мин)", color="#1f77b4", alpha=0.8)
    ax2.bar(x + 0.2, tde_vals, 0.35, label="|TDE| mean (мин)", color="#ff7f0e", alpha=0.8)
    for i, (m, t) in enumerate(zip(mae_vals, tde_vals)):
        ax2.text(i - 0.2, m + 0.03, f"{m:.2f}", ha="center", fontsize=8)
        ax2.text(i + 0.2, t + 0.03, f"{t:.2f}", ha="center", fontsize=8)
    ax2.set_xticks(x)
    ax2.set_xticklabels(versions, rotation=20, ha="right", fontsize=8)
    ax2.set_ylabel("мин")
    ax2.set_title("MAE и |TDE| по версиям")
    ax2.legend(fontsize=9)
    ax2.grid(alpha=0.3, axis="y")

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  → {out_path.relative_to(out_path.parents[1])}")


def plot_predictions_trajectories(
    df_pred: pd.DataFrame,
    title: str,
    out_path: Path,
) -> None:
    """Траектории предсказаний по участникам.

    Для каждого участника: y_true (пунктир) и y_pred (линия) в минутах
    по оси x — время от начала теста в минутах.
    Вертикальная линия — момент прохождения порога (y_true = 0).
    """
    subjects = sorted(df_pred["subject_id"].unique())
    n = len(subjects)
    ncols = min(4, n)
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4.5, nrows * 3.2),
                             squeeze=False)
    fig.suptitle(title, fontsize=12, fontweight="bold")

    for idx, subj in enumerate(subjects):
        ax = axes[idx // ncols][idx % ncols]
        d = df_pred[df_pred["subject_id"] == subj].sort_values("elapsed_sec")

        elapsed_min = d["elapsed_sec"].values / 60.0
        y_true_min  = d["y_true_sec"].values / 60.0
        y_pred_min  = d["y_pred_sec"].values / 60.0

        ax.plot(elapsed_min, y_true_min, color="black", lw=1.5,
                ls="--", label="Истина", zorder=3)
        ax.plot(elapsed_min, y_pred_min, color="#1f77b4", lw=1.5,
                label="Предсказание", zorder=2)

        # Если есть сырые предсказания (до Kalman)
        if "y_pred_raw_sec" in d.columns and not (d["y_pred_raw_sec"] == d["y_pred_sec"]).all():
            ax.plot(elapsed_min, d["y_pred_raw_sec"].values / 60.0,
                    color="#aec7e8", lw=1.0, ls=":", alpha=0.7, label="Raw")

        # Момент порога
        cross_idx = np.where(y_true_min <= 0)[0]
        if len(cross_idx) > 0:
            t_cross = elapsed_min[cross_idx[0]]
            ax.axvline(t_cross, color="red", lw=1.2, ls=":", alpha=0.7)

        ax.axhline(0, color="gray", lw=0.8, ls="-", alpha=0.4)
        ax.set_title(subj, fontsize=9)
        ax.set_xlabel("Время теста, мин", fontsize=7)
        ax.set_ylabel("τ, мин", fontsize=7)
        ax.tick_params(labelsize=7)
        ax.grid(alpha=0.25)

    # Легенда на первом субплоте
    axes[0][0].legend(fontsize=7, loc="upper right")

    # Скрыть пустые ячейки
    for idx in range(n, nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  → {out_path.relative_to(out_path.parents[2])}")


def plot_error_by_subject(
    df_pred: pd.DataFrame,
    title: str,
    out_path: Path,
) -> None:
    """Ошибки предсказания по участникам: boxplot + глобальное распределение."""
    subjects = sorted(df_pred["subject_id"].unique())
    errors_min = (df_pred["y_pred_sec"] - df_pred["y_true_sec"]) / 60.0

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(title, fontsize=12, fontweight="bold")

    # Boxplot по участникам
    data_per_subj = [errors_min[df_pred["subject_id"] == s].values for s in subjects]
    bp = ax1.boxplot(data_per_subj, labels=subjects, vert=False,
                     patch_artist=True, notch=False,
                     medianprops={"color": "red", "lw": 2})
    for patch in bp["boxes"]:
        patch.set_facecolor("#aec7e8")
        patch.set_alpha(0.7)
    ax1.axvline(0, color="black", lw=1.2)
    medians = [np.median(d) for d in data_per_subj]
    mean_all = float(errors_min.mean())
    ax1.axvline(mean_all, color="navy", lw=1.5, ls="--",
                label=f"Среднее {mean_all:+.2f} мин")
    ax1.set_xlabel("Ошибка, мин  (+ = модель завышает τ)", fontsize=9)
    ax1.set_title("Ошибки по участникам")
    ax1.legend(fontsize=9)
    ax1.grid(alpha=0.3, axis="x")

    # Гистограмма глобального распределения ошибок
    ax2.hist(errors_min.values, bins=40, color="#7f7f7f", alpha=0.75, edgecolor="white")
    ax2.axvline(0, color="black", lw=1.5)
    ax2.axvline(float(errors_min.mean()), color="red", lw=1.5, ls="--",
                label=f"μ={errors_min.mean():+.2f} мин")
    ax2.axvline(float(errors_min.median()), color="orange", lw=1.5, ls=":",
                label=f"median={errors_min.median():+.2f} мин")
    ax2.set_xlabel("Ошибка, мин", fontsize=9)
    ax2.set_ylabel("Окна", fontsize=9)
    ax2.set_title("Распределение ошибок (все окна)")
    ax2.legend(fontsize=9)
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  → {out_path.relative_to(out_path.parents[2])}")


def plot_mae_split(
    metrics: dict,
    title: str,
    out_path: Path,
) -> None:
    """MAE разбитый по зонам: до порога, после порога, вблизи порога."""
    labels = ["До порога\n(τ > 0)", "После порога\n(τ ≤ 0)", f"Вблизи порога\n(|τ| ≤ 2 мин)"]
    values = [metrics["mae_pre"], metrics["mae_post"], metrics["mae_near"]]
    counts = [metrics["n_pre"],  metrics["n_post"],  metrics["n_near"]]
    colors = ["#2ca02c", "#d62728", "#ff7f0e"]

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(labels, values, color=colors, alpha=0.8, edgecolor="white", width=0.5)

    # Аннотации: MAE и количество окон
    for bar, val, cnt in zip(bars, values, counts):
        if not np.isnan(val):
            ax.text(bar.get_x() + bar.get_width() / 2, val + 0.02,
                    f"{val:.3f} мин\n(n={cnt})",
                    ha="center", va="bottom", fontsize=10, fontweight="bold")

    # Глобальный MAE для сравнения
    ax.axhline(metrics["mae_min"], color="black", lw=1.5, ls="--",
               label=f"MAE global = {metrics['mae_min']:.3f} мин")

    ax.set_ylim(0, max(v for v in values if not np.isnan(v)) * 1.25)
    ax.set_ylabel("MAE, мин", fontsize=11)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3, axis="y")

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  → {out_path.relative_to(out_path.parents[2])}")


def plot_split_mae_comparison(
    all_metrics: dict[str, dict],
    out_path: Path,
    title: str = "MAE по зонам — сравнение версий",
) -> None:
    """Сравнительный график mae_pre / mae_post / mae_near по всем версиям."""
    versions = list(all_metrics.keys())
    x = np.arange(len(versions))
    width = 0.25
    colors = {"mae_pre": "#2ca02c", "mae_post": "#d62728", "mae_near": "#ff7f0e"}
    labels_ru = {"mae_pre": "До порога (τ>0)", "mae_post": "После порога (τ≤0)",
                 "mae_near": "Вблизи (|τ|≤2 мин)"}

    fig, ax = plt.subplots(figsize=(max(10, len(versions) * 1.5), 5))
    fig.suptitle(title, fontsize=12, fontweight="bold")

    for j, key in enumerate(["mae_pre", "mae_post", "mae_near"]):
        vals = [all_metrics[v].get(key, float("nan")) for v in versions]
        ax.bar(x + (j - 1) * width, vals, width, label=labels_ru[key],
               color=colors[key], alpha=0.8)
        for i, val in enumerate(vals):
            if not np.isnan(val):
                ax.text(i + (j - 1) * width, val + 0.02, f"{val:.2f}",
                        ha="center", va="bottom", fontsize=7, rotation=90)

    ax.set_xticks(x)
    ax.set_xticklabels(versions, rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("MAE, мин")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3, axis="y")

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  → {out_path.relative_to(out_path.parents[1])}")
