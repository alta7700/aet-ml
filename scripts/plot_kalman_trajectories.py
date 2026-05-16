"""Сводная панель 2x2: траектории предсказаний raw vs Kalman vs истина.

Иллюстрация к разделу 3.1 — почему знак выигрыша от Калмана различается:
Калман задерживает реакцию на изменения, поэтому помогает «дрожащим»
прогнозам и вредит «инерционным».

Ячейки:
  ┌──────────────────────────┬──────────────────────────┐
  │ LT1, LSTM (gain > 0)     │ LT1, Классика (gain < 0) │
  ├──────────────────────────┼──────────────────────────┤
  │ LT2, LSTM (gain « 0)     │ LT2, Ансамбль (gain ≈ 0) │
  └──────────────────────────┴──────────────────────────┘

Для каждой ячейки — один типичный субъект (медианный по MAE среди субъектов
данной конфигурации).

Источники:
- results/<version>/ypred_<tgt>_<fset>.npy, ytrue_<tgt>_<fset>.npy
- results/summary_all_versions.csv (best_sigma_obs)
- DEFAULT_DATASET_DIR/merged_features_ml.parquet (для маппинга окон → субъекты)

Выход: results/final/plots/kalman_trajectories_2x2.png
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

from dataset_pipeline.common import DEFAULT_DATASET_DIR  # noqa: E402

RESULTS = ROOT / "results"
OUT = RESULTS / "final" / "plots" / "kalman_trajectories_2x2.png"

# 4 конфигурации: (label-row, version, family-label, target, feature_set)
CELLS = [
    ("LT1 · Wavelet-LSTM\n(Калман улучшает)",      "v0103", "lt1", "EMG_NIRS_HRV"),
    ("LT1 · Классика (Ridge)\n(Калман ухудшает)",  "v0011", "lt1", "EMG_NIRS_HRV"),
    ("LT2 · LSTM\n(Калман сильно ухудшает)",       "v0101", "lt2", "EMG_NIRS_HRV"),
    ("LT2 · Ансамбль\n(Калман почти не влияет)",   "v0107", "lt2", "EMG_NIRS_HRV"),
]


def kalman_smooth(y_pred: np.ndarray, sigma_p: float, sigma_obs: float,
                  dt: float = 5.0) -> np.ndarray:
    """Одномерный фильтр Калмана; модель: x[t] = x[t-1] − dt."""
    x = float(y_pred[0])
    p = sigma_obs ** 2
    out = np.empty(len(y_pred))
    for i in range(len(y_pred)):
        x -= dt
        p += sigma_p ** 2
        k = p / (p + sigma_obs ** 2)
        x = x + k * (y_pred[i] - x)
        p = (1 - k) * p
        out[i] = x
    return out


def load_window_index(target: str) -> pd.DataFrame:
    """Воспроизводит фильтр и сортировку из prepare_data, чтобы получить
    позиционное соответствие индекс окна → (subject_id, window_start_sec)."""
    df = pd.read_parquet(DEFAULT_DATASET_DIR / "merged_features_ml.parquet")
    if target == "lt2":
        df = df[df["window_valid_all_required"] == 1].copy()
        df = df.dropna(subset=["target_time_to_lt2_center_sec"])
    else:
        df = df[df["target_time_to_lt1_usable"] == 1].copy()
    df = df.sort_values(["subject_id", "window_start_sec"]).reset_index(drop=True)
    return df[["subject_id", "window_start_sec"]]


def pick_median_subject(per_subj: dict[str, np.ndarray],
                        ytrue_per: dict[str, np.ndarray]) -> str:
    """Выбирает субъекта с MAE, ближайшим к медиане по группе."""
    maes = {s: np.mean(np.abs(per_subj[s] - ytrue_per[s])) / 60.0
            for s in per_subj}
    med = float(np.median(list(maes.values())))
    return min(maes, key=lambda s: abs(maes[s] - med))


def load_cell(version: str, target: str, fset_tag: str
              ) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray],
                          dict[str, np.ndarray], float]:
    """Возвращает per_subj raw/true/wstart и best_sigma_obs."""
    yp = np.load(RESULTS / version / f"ypred_{target}_{fset_tag}.npy")
    yt = np.load(RESULTS / version / f"ytrue_{target}_{fset_tag}.npy")
    idx = load_window_index(target)
    # Если ypred короче — версия дропнула отдельных субъектов.
    # Список оставшихся берём из per_subject.csv этой версии.
    if len(idx) != len(yp):
        ps = pd.read_csv(RESULTS / version / "per_subject.csv")
        ps = ps[(ps.target == target)
                & (ps.feature_set == fset_tag.replace("_", "+"))]
        kept = sorted(ps["subject_id"].unique())
        idx = idx[idx["subject_id"].isin(kept)].reset_index(drop=True)
        if len(idx) != len(yp):
            raise RuntimeError(
                f"не удалось сопоставить размеры для {version}/{target}/{fset_tag}: "
                f"index={len(idx)}, ypred={len(yp)}, kept={len(kept)}"
            )

    raw_per, true_per, ws_per = {}, {}, {}
    for s, sub in idx.groupby("subject_id"):
        mask = idx["subject_id"].values == s
        raw_per[s]  = yp[mask]
        true_per[s] = yt[mask]
        ws_per[s]   = sub["window_start_sec"].values

    summary = pd.read_csv(RESULTS / "summary_all_versions.csv")
    fset_csv = fset_tag.replace("_", "+")
    row = summary[(summary.version == version) & (summary.target == target)
                  & (summary.feature_set == fset_csv)
                  & (summary.variant == "with_abs")].iloc[0]
    return raw_per, true_per, ws_per, float(row["best_sigma_obs"])


def plot_panel(ax, raw: np.ndarray, true: np.ndarray, kal: np.ndarray,
               ws: np.ndarray, title: str) -> None:
    # Отбрасываем 3 первых окна — переходный режим Калмана с начальной σ_obs².
    skip = 3
    t = ws[skip:] / 60.0
    raw_min  = raw[skip:]  / 60.0
    true_min = true[skip:] / 60.0
    kal_min  = kal[skip:]  / 60.0
    ax.plot(t, true_min, color="black",  lw=1.8, label="истина", zorder=3)
    ax.plot(t, raw_min,  color="#1f77b4", lw=1.0, alpha=0.75,
            label="raw", zorder=2)
    ax.plot(t, kal_min,  color="#d62728", lw=1.6,
            label="Kalman", zorder=2)
    ax.axhline(0, color="gray", lw=0.5, ls="--")
    ax.set_title(title, fontsize=11)
    ax.grid(alpha=0.3)
    # Y-предел — по объединению всех трёх кривых.
    all_v = np.concatenate([raw_min, true_min, kal_min])
    lo, hi = float(all_v.min()), float(all_v.max())
    pad = (hi - lo) * 0.08
    ax.set_ylim(lo - pad, hi + pad)


def main() -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharex=False)
    axes = axes.flatten()

    for ax, (title, version, target, fset) in zip(axes, CELLS):
        raw_per, true_per, ws_per, sigma_obs = load_cell(version, target, fset)
        # Калман per subject с тем же best_sigma, что и в summary.
        kal_per = {s: kalman_smooth(raw_per[s], sigma_p=5.0, sigma_obs=sigma_obs)
                   for s in raw_per}
        subj = pick_median_subject(raw_per, true_per)
        plot_panel(ax, raw_per[subj], true_per[subj], kal_per[subj],
                   ws_per[subj],
                   f"{title}\n(субъект {subj}, σ_obs={sigma_obs:g})")

    # Общие подписи осей.
    for ax in axes[2:]:
        ax.set_xlabel("Время записи, мин")
    for ax in (axes[0], axes[2]):
        ax.set_ylabel("Время до порога, мин")

    # Единая легенда сверху.
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3,
               bbox_to_anchor=(0.5, 1.02), frameon=False, fontsize=10)

    fig.tight_layout(rect=(0, 0, 1, 0.97))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"saved: {OUT}")


if __name__ == "__main__":
    main()
