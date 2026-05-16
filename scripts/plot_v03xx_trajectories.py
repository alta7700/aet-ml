"""
Анализ результатов v03xx: поиск лучших моделей и построение траекторий.

Для каждой из топ-3 моделей строит 2 графика:
  1. Raw prediction — траектории всех субъектов без сглаживания
  2. Kalman prediction — те же траектории после фильтра Калмана (sigma_obs=30)

Запуск:
  python3 scripts/plot_v03xx_trajectories.py
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# ───────────────────── константы ─────────────────────
RESULTS_DIR = Path("/Users/tascan/Desktop/диссер/results")
OUT_DIR     = RESULTS_DIR / "v03xx_analysis"
OUT_DIR.mkdir(exist_ok=True)

# Доступные версии с полными данными (есть summary.csv)
VERSIONS = ["v0301", "v0302", "v0303", "v0304", "v0305", "v0306",
            "v0307", "v0308", "v0309", "v0310"]

DT     = 5.0   # шаг окна, сек
SIGMA_P   = 5.0
SIGMA_OBS = 30.0  # лучший sigma по результатам


# ───────────────────── Kalman-фильтр ─────────────────────
def kalman_smooth(y_pred: np.ndarray,
                  sigma_p: float = SIGMA_P,
                  sigma_obs: float = SIGMA_OBS) -> np.ndarray:
    """Одномерный фильтр Калмана (модель: x[t] = x[t-1] - dt)."""
    n  = len(y_pred)
    x  = float(y_pred[0])
    p  = sigma_obs ** 2
    smoothed = np.empty(n, dtype=np.float32)
    for i in range(n):
        x -= DT
        p += sigma_p ** 2
        k  = p / (p + sigma_obs ** 2)
        x  = x + k * (float(y_pred[i]) - x)
        p  = (1 - k) * p
        smoothed[i] = x
    return smoothed


# ───────────────────── разбиение на субъектов ─────────────────────
def split_by_subject(ytrue: np.ndarray, ypred: np.ndarray):
    """Разбивает массивы по субъектам, определяя границы по скачкам ytrue."""
    diffs = np.diff(ytrue)
    boundaries = np.where(diffs > 10)[0] + 1          # скачок > 10с = новый субъект
    splits = np.split(ytrue, boundaries), np.split(ypred, boundaries)
    return list(zip(splits[0], splits[1]))             # [(ytrue_s, ypred_s), ...]


# ───────────────────── сбор summary ─────────────────────
def collect_summary() -> pd.DataFrame:
    rows = []
    for ver in VERSIONS:
        f = RESULTS_DIR / ver / "summary.csv"
        if not f.exists():
            continue
        df = pd.read_csv(f)
        df["version"] = ver
        rows.append(df)
    return pd.concat(rows, ignore_index=True)


# ───────────────────── построение графиков ─────────────────────
def plot_trajectories(version: str, feature_set: str, target: str,
                      raw_mae: float, kalman_mae: float, r2: float, rho: float):
    """Строит 2 графика (raw + kalman) для одной модели.

    X-ось: ytrue (реальное время до порога, мин) — убывает слева направо.
    Y-ось: ypred (предсказанное время до порога, мин).
    Идеальная модель: y = x (диагональ).
    Такой формат выравнивает все субъекты по моменту LT (x=0) и корректен
    для любого типа модели (stateless / stateful / record-level).
    """
    fset_tag = feature_set.replace("+", "_")
    tag      = f"{target}_{fset_tag}"

    ypred_path = RESULTS_DIR / version / f"ypred_{tag}.npy"
    ytrue_path = RESULTS_DIR / version / f"ytrue_{tag}.npy"

    if not ypred_path.exists() or not ytrue_path.exists():
        print(f"  ⚠  Файлы не найдены: {version}/{tag}")
        return

    ypred_all = np.load(ypred_path)
    ytrue_all = np.load(ytrue_path)

    subjects = split_by_subject(ytrue_all, ypred_all)
    n_subj   = len(subjects)
    colors   = cm.tab20(np.linspace(0, 1, n_subj))

    label_target = "LT1" if target == "lt1" else "LT2"
    title_base   = (f"{version}  ·  {feature_set}  ·  {label_target}\n"
                    f"MAE_raw={raw_mae:.3f}s  MAE_kalman={kalman_mae:.3f}s  "
                    f"R²={r2:.3f}  ρ={rho:.3f}")

    # Общий диапазон по ytrue для диагонали
    all_yt = np.concatenate([yt for yt, _ in subjects]) / 60
    diag_min = max(all_yt.min() - 1, all_yt.min())
    diag_max = all_yt.max() + 1

    for mode in ("raw", "kalman"):
        fig, ax = plt.subplots(figsize=(11, 7))

        for idx, (yt, yp) in enumerate(subjects):
            xt   = yt / 60         # реальное время до порога, мин (x)
            yp_plot = kalman_smooth(yp) / 60 if mode == "kalman" else yp / 60

            ax.plot(xt, yp_plot,
                    color=colors[idx], alpha=0.75, linewidth=1.5,
                    label=f"S{idx+1:02d}", zorder=2)

            # Начальная точка (крестик)
            ax.scatter([xt[0]], [yp_plot[0]],
                       color=colors[idx], s=20, zorder=3, alpha=0.6)

        # Диагональ "идеальный детектор" (y = x)
        diag = np.array([diag_min, diag_max])
        ax.plot(diag, diag, "k--", linewidth=1.2, alpha=0.4, label="y = x (ideal)", zorder=1)

        # Вертикальная черта: момент порога
        ax.axvline(x=0, color="black", linewidth=1.0, alpha=0.3, linestyle=":")

        # Ось x убывает (тест движется слева направо — приближение к LT)
        ax.invert_xaxis()

        mode_label = "Kalman (σ=30с)" if mode == "kalman" else "Raw"
        ax.set_xlabel("Реальное время до порога (ytrue), мин  ←  тест идёт вправо", fontsize=11)
        ax.set_ylabel("Предсказанное время до порога, мин", fontsize=11)
        ax.set_title(f"{title_base}\n[{mode_label}]", fontsize=11)
        ax.legend(loc="lower right", fontsize=7, ncol=3, framealpha=0.6)
        ax.grid(True, alpha=0.3)

        fname = OUT_DIR / f"{version}_{tag}_{mode}.png"
        fig.tight_layout()
        fig.savefig(fname, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  💾 {fname.name}")


# ───────────────────── main ─────────────────────
def main():
    print("Собираем сводку по всем версиям...")
    summary = collect_summary()
    print(f"  Всего конфигураций: {len(summary)}")

    # Печатаем топ-10 по raw_mae_min
    top = summary.nsmallest(10, "raw_mae_min")[
        ["version", "feature_set", "target", "raw_mae_min", "kalman_mae_min", "r2", "rho"]
    ]
    print("\n📊 Топ-10 моделей по raw MAE:")
    print(top.to_string(index=False))

    # Сохраняем полную таблицу
    summary_path = OUT_DIR / "summary_all.csv"
    summary.to_csv(summary_path, index=False)
    print(f"\n  Полная таблица → {summary_path.name}")

    # Строим графики для топ-3
    top3 = summary.nsmallest(3, "raw_mae_min").reset_index(drop=True)
    print("\n🎨 Строим траектории для топ-3 моделей...")
    for _, row in top3.iterrows():
        print(f"\n  {row['version']}  {row['feature_set']}  {row['target']}  "
              f"MAE={row['raw_mae_min']:.3f}  R²={row['r2']:.3f}")
        plot_trajectories(
            version     = row["version"],
            feature_set = row["feature_set"],
            target      = row["target"],
            raw_mae     = row["raw_mae_min"],
            kalman_mae  = row["kalman_mae_min"],
            r2          = row["r2"],
            rho         = row["rho"],
        )

    print(f"\n✅ Готово. Графики в {OUT_DIR}")


if __name__ == "__main__":
    main()
