"""Шаг 02: единый ranker и Top-10 по нескольким критериям.

Читает:
- results/summary_all_versions.csv
- results/<version>/per_subject.csv  для всех 9 версий

Производит:
- results/final/01_ranking_wide.csv          — широкая таблица с производными
- results/final/02_topk_by_target_fset.md    — топы по (target, feature_set)
- results/final/03_topk_by_target_version.md — топы по (target, version)
- results/final/04_topk_by_target_fset_family.md — топы по (target, fset, family)
- results/final/05_candidates_intersection.csv — пересечение топов (группа A)
- results/final/plots/stability_vs_mae.png
- results/final/plots/kalman_gain_by_version.png
- results/final/README.md (создаётся/обновляется только своим разделом)

Запуск:
    PYTHONPATH=. uv run python scripts/final_build_ranking.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FINAL = RESULTS / "final"
PLOTS = FINAL / "plots"

VERSIONS = ["v0011", "v0101", "v0102", "v0103", "v0104", "v0105",
            "v0106a", "v0106b", "v0106c", "v0107"]

# Маппинг архитектурных семейств (для группировки C).
# Уточнять при необходимости — это рабочая гипотеза.
FAMILY_MAP = {
    "v0011": "linear",
    "v0101": "lstm",
    "v0104": "lstm",
    "v0102": "tcn",
    "v0105": "tcn",
    "v0103": "wavelet",
    "v0106a": "wavelet",
    "v0106b": "wavelet",
    "v0106c": "wavelet",
    "v0107": "ensemble",
}

# Количество позиций в топе.
K = 10

# Критерии (column_name, ascending, label).
CRITERIA = [
    ("raw_mae_min",      True,  "raw_mae_min ↑"),
    ("kalman_mae_min",   True,  "kalman_mae_min ↑"),
    ("loso_mae_median",  True,  "loso_mae_median ↑"),
    ("loso_mae_std",     True,  "loso_mae_std ↑ (стабильность)"),
    ("rho",              False, "rho ↓"),
    ("r2",               False, "r2 ↓"),
]


# ─────────────────────── Загрузка per-subject ───────────────────────

def load_per_subject_v0011() -> pd.DataFrame:
    """Загружает per-subject v0011, объединяя per_subject_full.csv (lt1+lt2 best inner)
    с per_subject.csv (для определения inner_model по lt2).

    Возврат: схема (version, target, feature_set, subject_id, mae_min, r2, inner_model).
    """
    full_path = RESULTS / "v0011" / "per_subject_full.csv"
    if full_path.exists():
        df_full = pd.read_csv(full_path)
        df_full["version"] = "v0011"
    else:
        df_full = pd.DataFrame()

    # Определяем inner_model для каждой (target, feature_set) из старого per_subject.csv
    # (там есть колонка 'model' с зоопарком). Берём по медиане MAE.
    old = pd.read_csv(RESULTS / "v0011" / "per_subject.csv")
    g = old.groupby(["target", "feature_set", "model"], as_index=False)["mae_min"].median()
    winners = (g.loc[g.groupby(["target", "feature_set"])["mae_min"].idxmin()]
                [["target", "feature_set", "model"]]
                .rename(columns={"model": "inner_model"}))

    if df_full.empty:
        # Fallback на старый: использовать выборку winning model.
        df = old.merge(winners.rename(columns={"inner_model": "model"}),
                       on=["target", "feature_set", "model"])
        df["version"] = "v0011"
        df["inner_model"] = df["model"]
        return df[["version", "target", "feature_set", "subject_id",
                   "mae_min", "r2", "inner_model"]]

    out = df_full.merge(winners, on=["target", "feature_set"], how="left")
    return out[["version", "target", "feature_set", "subject_id",
                "mae_min", "r2", "inner_model"]]


def load_per_subject_nn(version: str) -> pd.DataFrame:
    """NN-схема: (variant, feature_set, target, subject_id, mae_min, r2).

    Приоритет: per_subject_full.csv (содержит lt1+lt2). Если его нет —
    fallback на per_subject.csv.
    """
    full_path = RESULTS / version / "per_subject_full.csv"
    if full_path.exists():
        df = pd.read_csv(full_path)
    else:
        df = pd.read_csv(RESULTS / version / "per_subject.csv")
        if version == "v0107":
            df["mae_min"] = df["mae_ens_min"]
            df["r2"] = np.nan
    df["version"] = version
    df["inner_model"] = "ensemble" if version == "v0107" else ""
    return df[["version", "variant", "target", "feature_set", "subject_id",
               "mae_min", "r2", "inner_model"]]


# ─────────────────────── Агрегаты по per-subject ───────────────────────

def aggregate_per_subject(ps: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    """Считает loso-агрегаты по группам."""
    rows = []
    for keys, sub in ps.groupby(group_cols):
        mae = sub["mae_min"].astype(float)
        r2 = sub["r2"].astype(float)
        row = dict(zip(group_cols, keys if isinstance(keys, tuple) else (keys,)))
        row["loso_n_subjects"] = int(mae.notna().sum())
        row["loso_mae_mean"]   = round(float(mae.mean()), 4)
        row["loso_mae_median"] = round(float(mae.median()), 4)
        row["loso_mae_std"]    = round(float(mae.std(ddof=1)) if len(mae) > 1 else float("nan"), 4)
        row["loso_mae_iqr"]    = round(float(mae.quantile(0.75) - mae.quantile(0.25)), 4)
        row["loso_r2_median"]  = round(float(r2.median()), 3)
        row["loso_neg_r2_share"] = round(float((r2 < 0).mean()), 3)
        # inner_model (только для v0011)
        if "inner_model" in sub.columns:
            uniq = sub["inner_model"].dropna().unique()
            row["inner_model"] = uniq[0] if len(uniq) == 1 else ""
        rows.append(row)
    return pd.DataFrame(rows)


# ─────────────────────── Сборка широкой таблицы ───────────────────────

def build_ranking_wide() -> pd.DataFrame:
    summary = pd.read_csv(RESULTS / "summary_all_versions.csv")
    summary["family"] = summary["version"].map(FAMILY_MAP)

    # per-subject объединение всех версий
    nn_frames = [load_per_subject_nn(v) for v in VERSIONS if v != "v0011"]
    v11 = load_per_subject_v0011()

    # Агрегаты NN: по (version, variant, target, feature_set).
    agg_nn = aggregate_per_subject(
        pd.concat(nn_frames, ignore_index=True),
        ["version", "variant", "target", "feature_set"],
    )
    # Агрегаты v0011: по (version, target, feature_set). Variant не различается.
    agg_v11 = aggregate_per_subject(
        v11.assign(version="v0011"),
        ["version", "target", "feature_set"],
    )

    # Мерж с summary
    nn_merge = summary[summary["version"] != "v0011"].merge(
        agg_nn, on=["version", "variant", "target", "feature_set"], how="left"
    )
    v11_merge = summary[summary["version"] == "v0011"].merge(
        agg_v11, on=["version", "target", "feature_set"], how="left"
    )
    if "inner_model" not in nn_merge.columns:
        nn_merge["inner_model"] = ""

    wide = pd.concat([nn_merge, v11_merge], ignore_index=True)

    # Производные
    wide["gap_mae"]     = (wide["loso_mae_mean"] - wide["raw_mae_min"]).round(4)
    wide["kalman_gain"] = (wide["raw_mae_min"] - wide["kalman_mae_min"]).round(4)

    # Упорядочим колонки
    front = ["version", "family", "variant", "target", "feature_set", "inner_model",
             "raw_mae_min", "kalman_mae_min", "kalman_30", "kalman_50", "kalman_75", "kalman_150",
             "r2", "rho", "n_windows",
             "loso_n_subjects", "loso_mae_mean", "loso_mae_median",
             "loso_mae_std", "loso_mae_iqr", "loso_r2_median", "loso_neg_r2_share",
             "gap_mae", "kalman_gain"]
    front = [c for c in front if c in wide.columns]
    rest = [c for c in wide.columns if c not in front]
    wide = wide[front + rest]
    return wide


# ─────────────────────── Топы и пересечения ───────────────────────

def model_id(row: pd.Series) -> str:
    """Короткий идентификатор модели для таблиц."""
    parts = [row["version"], row["variant"], row["target"], row["feature_set"]]
    return " | ".join(str(p) for p in parts)


def topk_in_group(df: pd.DataFrame, k: int) -> dict[str, pd.DataFrame]:
    """Для каждого критерия возвращает топ-K из df (внутри одной группы)."""
    tops = {}
    for col, asc, label in CRITERIA:
        sub = df[df[col].notna()].copy()
        if sub.empty:
            tops[label] = sub
            continue
        sub = sub.sort_values(col, ascending=asc).head(k)
        tops[label] = sub
    return tops


def write_topk_md(path: Path, wide: pd.DataFrame, group_cols: list[str], k: int) -> None:
    lines: list[str] = []
    lines.append(f"# Top-{k} по группам: {' × '.join(group_cols)}\n")
    lines.append(f"Критериев: {len(CRITERIA)}. ↑ — меньше лучше, ↓ — больше лучше.\n")

    for keys, sub in wide.groupby(group_cols):
        keys_t = keys if isinstance(keys, tuple) else (keys,)
        header = " | ".join(f"{c}={v}" for c, v in zip(group_cols, keys_t))
        lines.append(f"\n## {header}  (n={len(sub)})\n")
        tops = topk_in_group(sub, k)
        for col, asc, label in CRITERIA:
            t = tops[label]
            lines.append(f"\n### {label}\n")
            if t.empty:
                lines.append("_нет данных_\n")
                continue
            show_cols = ["version", "variant", "feature_set", "target",
                         "loso_mae_median", "loso_mae_std", "raw_mae_min",
                         "kalman_mae_min", "rho", "r2", "loso_neg_r2_share",
                         "loso_n_subjects", "inner_model"]
            show_cols = [c for c in show_cols if c in t.columns and c not in group_cols]
            lines.append(t[show_cols].to_markdown(index=False))
            lines.append("\n")
    path.write_text("\n".join(lines), encoding="utf-8")


def intersection_candidates(wide: pd.DataFrame, group_cols: list[str], k: int,
                             min_hits: int) -> pd.DataFrame:
    """Для группы A: модель попала в Top-K ≥ min_hits критериев → кандидат."""
    rows = []
    for keys, sub in wide.groupby(group_cols):
        keys_t = keys if isinstance(keys, tuple) else (keys,)
        tops = topk_in_group(sub, k)
        for _, row in sub.iterrows():
            mid = model_id(row)
            hits = []
            for col, asc, label in CRITERIA:
                t = tops[label]
                if t.empty:
                    continue
                if mid in t.apply(model_id, axis=1).tolist():
                    hits.append(label)
            if len(hits) >= min_hits:
                rec = {c: v for c, v in zip(group_cols, keys_t)}
                rec.update({
                    "version": row["version"],
                    "variant": row["variant"],
                    "feature_set": row["feature_set"],
                    "target": row["target"],
                    "family": row.get("family", ""),
                    "inner_model": row.get("inner_model", ""),
                    "n_hits": len(hits),
                    "hit_criteria": "; ".join(hits),
                    "loso_mae_median": row["loso_mae_median"],
                    "loso_mae_std": row["loso_mae_std"],
                    "raw_mae_min": row["raw_mae_min"],
                    "kalman_mae_min": row["kalman_mae_min"],
                    "rho": row["rho"],
                    "r2": row["r2"],
                    "loso_n_subjects": row["loso_n_subjects"],
                })
                rows.append(rec)
    out = pd.DataFrame(rows)
    if not out.empty:
        # дедуп по (group + model_id) уже есть, сортируем
        out = out.sort_values(group_cols + ["n_hits"], ascending=[True] * len(group_cols) + [False])
    return out


# ─────────────────────── Графики ───────────────────────

def plot_stability_vs_mae(wide: pd.DataFrame, candidates: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=False)
    for ax, tgt in zip(axes, ["lt1", "lt2"]):
        sub = wide[wide["target"] == tgt].dropna(subset=["loso_mae_median", "loso_mae_std"])
        ax.scatter(sub["loso_mae_median"], sub["loso_mae_std"],
                   c="lightgray", s=30, label="all")
        # Помечаем кандидатов
        cand_sub = candidates[candidates["target"] == tgt] if not candidates.empty else candidates
        if not cand_sub.empty:
            ax.scatter(cand_sub["loso_mae_median"], cand_sub["loso_mae_std"],
                       c="C3", s=50, label="кандидаты (≥3 топа)")
            for _, r in cand_sub.iterrows():
                lbl = f"{r['version']}\n{r['feature_set']}"
                ax.annotate(lbl, (r["loso_mae_median"], r["loso_mae_std"]),
                            fontsize=7, alpha=0.7, xytext=(3, 3),
                            textcoords="offset points")
        ax.set_xlabel("loso_mae_median, мин")
        ax.set_ylabel("loso_mae_std, мин (нестабильность)")
        ax.set_title(f"{tgt.upper()}")
        ax.grid(alpha=0.3)
        ax.legend(loc="best", fontsize=8)
    fig.suptitle("Стабильность vs точность (LOSO)")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_kalman_gain(wide: pd.DataFrame, path: Path) -> None:
    g = (wide.dropna(subset=["kalman_gain"])
              .groupby(["version", "target"])["kalman_gain"].mean()
              .unstack("target"))
    fig, ax = plt.subplots(figsize=(10, 5))
    g.plot.bar(ax=ax)
    ax.axhline(0, color="black", lw=0.6)
    ax.set_ylabel("kalman_gain = raw_mae − kalman_mae, мин")
    ax.set_title("Средний выигрыш от Калмана по версиям")
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


# ─────────────────────── README ───────────────────────

README_TEXT = """# results/final/

Финальные артефакты для диссертации. Создаются скриптами из ../scripts/final_*.py.

## Шаг 02 — ранжирование моделей

- **01_ranking_wide.csv** — широкая таблица по всем 138 строкам summary с
  производными метриками LOSO (loso_mae_median/std/iqr, gap, kalman_gain и т.д.).
- **02_topk_by_target_fset.md** — главные топы: внутри каждой `(target, feature_set)`
  10 лучших по 6 критериям. Это «яблоки с яблоками».
- **03_topk_by_target_version.md** — внутри `(target, version)`: какая модальность
  лучше для каждой архитектуры.
- **04_topk_by_target_fset_family.md** — внутри `(target, feature_set, family)`:
  какая архитектура лучше при фиксированной модальности.
- **05_candidates_intersection.csv** — модели, попавшие в Top-10 одновременно по
  ≥3 из 6 критериев в группе (target, feature_set). Это шорт-лист для шага 04.

### Критерии Top-K
- `raw_mae_min` ↑   — MAE по всем окнам (взвешена, техническая)
- `kalman_mae_min` ↑ — MAE после калмановского сглаживания
- `loso_mae_median` ↑ — медиана MAE по субъектам LOSO (главная)
- `loso_mae_std` ↑   — стабильность по субъектам
- `rho` ↓           — Spearman, ранговое качество
- `r2` ↓            — R²

### Графики
- plots/stability_vs_mae.png — scatter точность vs стабильность, кандидаты подписаны.
- plots/kalman_gain_by_version.png — средний выигрыш от Калмана по версиям.

## Важные оговорки
- В v0011 per_subject имеет колонку `model` (зоопарк sklearn), variant не различается.
  Для агрегатов LOSO выбирается лучший внутренний model по медиане MAE на (target, fset);
  его имя в колонке `inner_model`.
- per_subject у NN-версий неполный: v0101=18, v0103–v0107=17, v0102/v0105=16 субъектов.
  Колонка `loso_n_subjects` — фактическое число.
"""


def main() -> None:
    FINAL.mkdir(parents=True, exist_ok=True)
    PLOTS.mkdir(parents=True, exist_ok=True)

    print("[1/5] Сборка широкой таблицы…")
    wide = build_ranking_wide()
    wide.to_csv(FINAL / "01_ranking_wide.csv", index=False)
    print(f"      → 01_ranking_wide.csv ({wide.shape})")

    print("[2/5] Топы по (target, feature_set)…")
    write_topk_md(FINAL / "02_topk_by_target_fset.md", wide,
                  ["target", "feature_set"], K)

    print("[3/5] Топы по (target, version)…")
    write_topk_md(FINAL / "03_topk_by_target_version.md", wide,
                  ["target", "version"], K)

    print("[4/5] Топы по (target, feature_set, family)…")
    write_topk_md(FINAL / "04_topk_by_target_fset_family.md", wide,
                  ["target", "feature_set", "family"], K)

    print("[5/5] Пересечения (группа A: target × feature_set, ≥3 из 6)…")
    cand = intersection_candidates(wide, ["target", "feature_set"], K, min_hits=3)
    cand.to_csv(FINAL / "05_candidates_intersection.csv", index=False)
    print(f"      → 05_candidates_intersection.csv ({len(cand)} строк)")

    print("[plots] stability_vs_mae, kalman_gain")
    plot_stability_vs_mae(wide, cand, PLOTS / "stability_vs_mae.png")
    plot_kalman_gain(wide, PLOTS / "kalman_gain_by_version.png")

    readme_path = FINAL / "README.md"
    if not readme_path.exists():
        readme_path.write_text(README_TEXT, encoding="utf-8")
    print(f"\nГотово. Все файлы в {FINAL}")


if __name__ == "__main__":
    main()
