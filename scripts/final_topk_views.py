"""Шаг 02 (доп.): узкие Top-5 по разным разрезам.

Читает results/final/01_ranking_wide.csv и строит:
- 06_top5_by_target_variant.md — Top-5 в каждой из 4 групп (target × variant):
    lt1_with_abs, lt1_noabs, lt2_with_abs, lt2_noabs.
- 07_top5_by_target_modality.md — Top-5 в каждой (target × feature_set);
    варианты with_abs/noabs остаются как самостоятельные строки.

Запуск:
    PYTHONPATH=. uv run python scripts/final_topk_views.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FINAL = ROOT / "results" / "final"
WIDE = FINAL / "01_ranking_wide.csv"

K = 5

# Те же критерии, что и в шаге 02.
CRITERIA = [
    ("raw_mae_min",      True,  "raw_mae_min ↑"),
    ("kalman_mae_min",   True,  "kalman_mae_min ↑"),
    ("loso_mae_median",  True,  "loso_mae_median ↑"),
    ("loso_mae_std",     True,  "loso_mae_std ↑ (стабильность)"),
    ("rho",              False, "rho ↓"),
    ("r2",               False, "r2 ↓"),
]

# Колонки, которые показываем в каждой Top-5 таблице (порядок важен).
SHOW_COLS = [
    "version", "variant", "feature_set", "family", "inner_model",
    "loso_mae_median", "loso_mae_std", "raw_mae_min", "kalman_mae_min",
    "rho", "r2", "loso_neg_r2_share", "loso_n_subjects",
]


def topk(df: pd.DataFrame, col: str, asc: bool, k: int) -> pd.DataFrame:
    """Top-k строк df по колонке col."""
    sub = df[df[col].notna()].copy()
    if sub.empty:
        return sub
    return sub.sort_values(col, ascending=asc).head(k)


def write_views(wide: pd.DataFrame, group_cols: list[str], out_path: Path,
                title: str, drop_in_table: list[str]) -> None:
    """Универсальный writer Top-K по группам.

    drop_in_table — какие колонки исключить из показа (как правило, ключи группы).
    """
    lines: list[str] = []
    lines.append(f"# {title}\n")
    lines.append(f"K = {K}. Критериев: {len(CRITERIA)}. ↑ — меньше лучше, ↓ — больше лучше.\n")

    for keys, sub in wide.groupby(group_cols):
        keys_t = keys if isinstance(keys, tuple) else (keys,)
        header = " | ".join(f"{c}={v}" for c, v in zip(group_cols, keys_t))
        lines.append(f"\n## {header}  (n={len(sub)})\n")
        for col, asc, label in CRITERIA:
            t = topk(sub, col, asc, K)
            lines.append(f"\n### {label}\n")
            if t.empty:
                lines.append("_нет данных_\n")
                continue
            show = [c for c in SHOW_COLS if c in t.columns and c not in drop_in_table]
            lines.append(t[show].to_markdown(index=False))
            lines.append("\n")

    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"  → {out_path.name}")


def main() -> None:
    wide = pd.read_csv(WIDE)
    print(f"Загружено {WIDE.name}: {wide.shape}")

    # (06) Top-5 по (target, variant)
    write_views(
        wide,
        group_cols=["target", "variant"],
        out_path=FINAL / "06_top5_by_target_variant.md",
        title="Top-5 по (target, variant) — 4 группы",
        drop_in_table=["target", "variant"],
    )

    # (07) Top-5 по (target, feature_set)
    write_views(
        wide,
        group_cols=["target", "feature_set"],
        out_path=FINAL / "07_top5_by_target_modality.md",
        title="Top-5 по модальностям (target × feature_set), варианты видны как строки",
        drop_in_table=["target", "feature_set"],
    )

    print("Готово.")


if __name__ == "__main__":
    main()
