"""Bar-plot «модальность × MAE_median» для варианта noabs.

В качестве компаньона к Рисунку 3.2 (with_abs) строит две панели:
левая — with_abs (как в Таблице 3.2), правая — noabs. Это даёт
визуальное обоснование выводов подраздела 3.4 (роль абсолютных
признаков для линейного семейства).

Данные — results/final/per_subject_all.csv (per-subject MAE по всем
версиям и обоим вариантам). Для каждой ячейки (variant, target, fset)
берётся чемпион — версия с наименьшей субъект-взвешенной медианной MAE.

Запуск:
    PYTHONPATH=. uv run python scripts/final_plot_modality_bar_noabs.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PS_PATH = ROOT / "results" / "final" / "per_subject_all.csv"
OUT_TBL = ROOT / "results" / "final" / "dissertation_tables" / "table_by_modality_noabs.csv"
OUT_PNG = ROOT / "results" / "final" / "plots" / "modality_vs_mae_noabs.png"

ORDER = ["EMG", "NIRS", "HRV", "EMG+NIRS", "EMG+NIRS+HRV"]


def champion_table(ps: pd.DataFrame) -> pd.DataFrame:
    """По каждой ячейке (variant, target, feature_set) выбирает версию
    с наименьшей субъект-взвешенной медианной MAE."""
    agg = (ps.groupby(["variant", "target", "feature_set", "version"])["mae_min"]
           .median().reset_index())
    agg = agg.sort_values(["variant", "target", "feature_set", "mae_min"])
    champ = agg.groupby(["variant", "target", "feature_set"]).head(1).reset_index(drop=True)
    return champ.rename(columns={"version": "best_version",
                                 "mae_min": "loso_mae_median_min"})


def draw_panel(ax, lt1: pd.Series, lt2: pd.Series, title: str, ymax: float) -> None:
    x = np.arange(len(ORDER))
    width = 0.36
    b1 = ax.bar(x - width / 2, lt1.values, width, label="LT1", color="#5B9BD5")
    b2 = ax.bar(x + width / 2, lt2.values, width, label="LT2", color="#ED7D31")

    lt1_min = float(np.nanmin(lt1.values))
    lt2_min = float(np.nanmin(lt2.values))
    for i, v in enumerate(lt1.values):
        if np.isfinite(v) and v == lt1_min:
            b1[i].set_edgecolor("black")
            b1[i].set_linewidth(2)
    for i, v in enumerate(lt2.values):
        if np.isfinite(v) and v == lt2_min:
            b2[i].set_edgecolor("black")
            b2[i].set_linewidth(2)

    for bars, vals in [(b1, lt1.values), (b2, lt2.values)]:
        for bar, v in zip(bars, vals):
            if np.isfinite(v):
                ax.text(bar.get_x() + bar.get_width() / 2, v + 0.05,
                        f"{v:.2f}", ha="center", va="bottom", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(ORDER)
    ax.set_xlabel("Модальность (набор признаков)")
    ax.set_ylabel("Субъект-взвешенная медианная MAE, мин")
    ax.set_title(title)
    ax.grid(alpha=0.3, axis="y")
    ax.legend(loc="upper right")
    ax.set_ylim(0, ymax)


def main() -> None:
    ps = pd.read_csv(PS_PATH)
    champ = champion_table(ps)

    # Сохраняем noabs-таблицу для текста (аналог table_by_modality.csv)
    noabs_tbl = champ[champ["variant"] == "noabs"][["target", "feature_set",
                                                     "best_version",
                                                     "loso_mae_median_min"]]
    noabs_tbl = noabs_tbl.sort_values(["target", "feature_set"])
    OUT_TBL.parent.mkdir(parents=True, exist_ok=True)
    noabs_tbl.to_csv(OUT_TBL, index=False)
    print(f"→ {OUT_TBL.name} ({len(noabs_tbl)} строк)")

    # Извлекаем серии для рисования
    def get_series(variant: str, target: str) -> pd.Series:
        sub = champ[(champ["variant"] == variant) & (champ["target"] == target)]
        return sub.set_index("feature_set")["loso_mae_median_min"].reindex(ORDER)

    wa_lt1 = get_series("with_abs", "lt1")
    wa_lt2 = get_series("with_abs", "lt2")
    na_lt1 = get_series("noabs", "lt1")
    na_lt2 = get_series("noabs", "lt2")

    ymax = max(float(np.nanmax([wa_lt1.max(), wa_lt2.max(),
                                 na_lt1.max(), na_lt2.max()])), 0.0) * 1.18

    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5), sharey=True)
    draw_panel(axes[0], wa_lt1, wa_lt2,
               "with_abs (Таблица 3.2): чемпион по модальностям", ymax)
    draw_panel(axes[1], na_lt1, na_lt2,
               "noabs: чемпион по модальностям", ymax)
    fig.suptitle("Лучшая модель линейного семейства по модальностям и вариантам\n"
                 "(LOSO, n = 18; жирная рамка — победитель в таргете)",
                 fontsize=11)
    fig.tight_layout()
    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG, dpi=130)
    plt.close(fig)
    print(f"→ {OUT_PNG}")


if __name__ == "__main__":
    main()
