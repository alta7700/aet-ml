"""Генерация bar-plot «модальность × MAE_median» для подраздела 3.2.

Данные — results/final/dissertation_tables/table_by_modality.csv.
Выход — results/final/plots/modality_vs_mae.png.

Запуск:
    uv run python scripts/final_plot_modality_bar.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TBL = ROOT / "results" / "final" / "dissertation_tables" / "table_by_modality.csv"
OUT = ROOT / "results" / "final" / "plots" / "modality_vs_mae.png"


def main() -> None:
    df = pd.read_csv(TBL)
    # Порядок модальностей — фиксированный, по возрастанию количества признаков
    order = ["EMG", "NIRS", "HRV", "EMG+NIRS", "EMG+NIRS+HRV"]
    df["fset_order"] = df["feature_set"].map({m: i for i, m in enumerate(order)})
    df = df.sort_values(["fset_order", "target"]).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(10, 5.5))
    x = np.arange(len(order))
    width = 0.36

    lt1 = df[df["target"] == "lt1"].set_index("feature_set").reindex(order)["loso_mae_median_min"]
    lt2 = df[df["target"] == "lt2"].set_index("feature_set").reindex(order)["loso_mae_median_min"]

    b1 = ax.bar(x - width / 2, lt1.values, width, label="LT1", color="#5B9BD5")
    b2 = ax.bar(x + width / 2, lt2.values, width, label="LT2", color="#ED7D31")

    # Подсветка чемпиона по каждому таргету
    lt1_min = float(lt1.min())
    lt2_min = float(lt2.min())
    for i, v in enumerate(lt1.values):
        if v == lt1_min:
            b1[i].set_edgecolor("black")
            b1[i].set_linewidth(2)
    for i, v in enumerate(lt2.values):
        if v == lt2_min:
            b2[i].set_edgecolor("black")
            b2[i].set_linewidth(2)

    for bars, vals in [(b1, lt1.values), (b2, lt2.values)]:
        for bar, v in zip(bars, vals):
            if np.isfinite(v):
                ax.text(bar.get_x() + bar.get_width() / 2, v + 0.05,
                        f"{v:.2f}", ha="center", va="bottom", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(order)
    ax.set_xlabel("Модальность (набор признаков)")
    ax.set_ylabel("Субъект-взвешенная медианная MAE, мин")
    ax.set_title("Лучшая модель линейного семейства по модальностям\n"
                 "(LOSO, n = 18; жирная рамка — победитель в таргете)")
    ax.grid(alpha=0.3, axis="y")
    ax.legend(loc="upper right")
    ax.set_ylim(0, max(lt1.max(), lt2.max()) * 1.18)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(OUT, dpi=130)
    plt.close(fig)
    print(f"→ {OUT}")


if __name__ == "__main__":
    main()
