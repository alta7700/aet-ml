"""Визуализации для слайдов «3 модальности».

Для каждого участника строит три фигуры с траекториями пер-оконных
признаков во времени и вертикальными проекциями LT1/LT2:

  1. DFA α1            -> figures/slides/dfa/<subject>.pdf|png
  2. NIRS SmO2 + HbDiff -> figures/slides/nirs/<subject>.pdf|png
  3. EMG RMS + MDF      -> figures/slides/emg/<subject>.pdf|png

На графиках только оси и легенда, без заголовков. Подписи русские.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ─────────────────────── Пути ───────────────────────

ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = ROOT / "dataset"
OUT_ROOT = ROOT / "analysis_out" / "figures" / "slides"


# ─────────────────────── Загрузка таблиц ───────────────────────

def load_tables() -> dict[str, pd.DataFrame]:
    """Загружает все нужные таблицы один раз."""
    return {
        "windows": pd.read_parquet(DATASET_DIR / "windows.parquet"),
        "hrv": pd.read_parquet(DATASET_DIR / "features_hrv.parquet"),
        "nirs": pd.read_parquet(DATASET_DIR / "features_nirs.parquet"),
        "emg": pd.read_parquet(DATASET_DIR / "features_emg_kinematics.parquet"),
        "subjects": pd.read_parquet(DATASET_DIR / "subjects.parquet"),
        "lt1": pd.read_parquet(DATASET_DIR / "lt1_labels.parquet"),
    }


# ─────────────────────── Утилиты ───────────────────────

def get_lt_times(subject_id: str, subjects: pd.DataFrame, lt1: pd.DataFrame) -> tuple[float | None, float | None]:
    """Возвращает (lt1_time_sec, lt2_time_sec) для участника. None — если недоступно."""
    lt1_row = lt1[lt1["subject_id"] == subject_id]
    lt1_time: float | None = None
    if not lt1_row.empty and int(lt1_row["lt1_available"].iloc[0]) == 1:
        # Сначала пробуем PCHIP-оценку, затем ступенчатую
        val = lt1_row["lt1_pchip_time_sec"].iloc[0]
        if pd.isna(val):
            val = lt1_row["lt1_time_sec"].iloc[0]
        lt1_time = float(val) if pd.notna(val) else None

    sub_row = subjects[subjects["subject_id"] == subject_id]
    lt2_time: float | None = None
    if not sub_row.empty:
        # Приоритет: refined → pchip → center
        for col in ("lt2_refined_time_sec", "lt2_pchip_time_sec", "lt2_time_center_sec"):
            val = sub_row[col].iloc[0]
            if pd.notna(val):
                lt2_time = float(val)
                break
    return lt1_time, lt2_time


def build_subject_frame(
    subject_id: str,
    windows: pd.DataFrame,
    features: pd.DataFrame,
    feature_cols: list[str],
) -> pd.DataFrame:
    """Соединяет окна с признаками и возвращает DataFrame с колонкой t_min."""
    w = windows[windows["subject_id"] == subject_id][["window_id", "window_center_sec"]]
    f = features[features["subject_id"] == subject_id][["window_id", *feature_cols]]
    merged = w.merge(f, on="window_id", how="inner").sort_values("window_center_sec")
    merged["t_min"] = merged["window_center_sec"] / 60.0
    return merged


def _draw_lt_lines(ax: plt.Axes, lt1_min: float | None, lt2_min: float | None) -> None:
    """Рисует вертикальные пунктиры LT1/LT2 на оси."""
    if lt1_min is not None:
        ax.axvline(lt1_min, color="#1f77b4", linestyle="--", linewidth=1.4, label="LT1")
    if lt2_min is not None:
        ax.axvline(lt2_min, color="#d62728", linestyle="--", linewidth=1.4, label="LT2")


def _save(fig: plt.Figure, out_dir: Path, subject_id: str) -> None:
    """Сохраняет фигуру в pdf и png."""
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_dir / f"{subject_id}.pdf", bbox_inches="tight")
    fig.savefig(out_dir / f"{subject_id}.png", bbox_inches="tight", dpi=160)
    plt.close(fig)


# ─────────────────────── DFA α1 ───────────────────────

def plot_dfa(subject_id: str, tables: dict[str, pd.DataFrame]) -> bool:
    """Строит траекторию DFA α1 с проекциями LT1/LT2."""
    df = build_subject_frame(subject_id, tables["windows"], tables["hrv"], ["hrv_dfa_alpha1"])
    valid = df.dropna(subset=["hrv_dfa_alpha1"])
    if valid.empty:
        return False

    lt1_s, lt2_s = get_lt_times(subject_id, tables["subjects"], tables["lt1"])
    lt1_min = lt1_s / 60.0 if lt1_s is not None else None
    lt2_min = lt2_s / 60.0 if lt2_s is not None else None

    fig, ax = plt.subplots(figsize=(6.5, 3.0))
    ax.plot(valid["t_min"], valid["hrv_dfa_alpha1"], color="#333333", linewidth=1.6, label="DFA α1")
    # Эталонный уровень 0.75 — переход aerobic → anaerobic (часто упоминается как порог)
    ax.axhline(0.75, color="gray", linestyle=":", linewidth=1.0, alpha=0.7, label="α1 = 0.75")
    _draw_lt_lines(ax, lt1_min, lt2_min)

    ax.set_xlabel("Время, мин")
    ax.set_ylabel("DFA α1")
    ax.legend(loc="best", frameon=False, fontsize=9)
    ax.grid(True, alpha=0.25)

    _save(fig, OUT_ROOT / "dfa", subject_id)
    return True


# ─────────────────────── NIRS ───────────────────────

def plot_nirs(subject_id: str, tables: dict[str, pd.DataFrame]) -> bool:
    """Строит траектории SmO2 и HbDiff (два стэка) с проекциями LT1/LT2."""
    df = build_subject_frame(
        subject_id,
        tables["windows"],
        tables["nirs"],
        ["trainred_smo2_mean", "trainred_hbdiff_mean"],
    )
    valid = df.dropna(subset=["trainred_smo2_mean", "trainred_hbdiff_mean"], how="all")
    if valid.empty:
        return False

    lt1_s, lt2_s = get_lt_times(subject_id, tables["subjects"], tables["lt1"])
    lt1_min = lt1_s / 60.0 if lt1_s is not None else None
    lt2_min = lt2_s / 60.0 if lt2_s is not None else None

    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(6.5, 4.2), sharex=True)

    ax_top.plot(valid["t_min"], valid["trainred_smo2_mean"], color="#2ca02c", linewidth=1.6, label="SmO₂")
    _draw_lt_lines(ax_top, lt1_min, lt2_min)
    ax_top.set_ylabel("SmO₂, %")
    ax_top.legend(loc="best", frameon=False, fontsize=9)
    ax_top.grid(True, alpha=0.25)

    ax_bot.plot(valid["t_min"], valid["trainred_hbdiff_mean"], color="#9467bd", linewidth=1.6, label="HbDiff")
    _draw_lt_lines(ax_bot, lt1_min, lt2_min)
    ax_bot.set_xlabel("Время, мин")
    ax_bot.set_ylabel("HbDiff, отн. ед.")
    ax_bot.legend(loc="best", frameon=False, fontsize=9)
    ax_bot.grid(True, alpha=0.25)

    _save(fig, OUT_ROOT / "nirs", subject_id)
    return True


# ─────────────────────── EMG ───────────────────────

def plot_emg(subject_id: str, tables: dict[str, pd.DataFrame]) -> bool:
    """Строит траектории RMS и медианной частоты (VL-дист, фаза load)."""
    df = build_subject_frame(
        subject_id,
        tables["windows"],
        tables["emg"],
        ["vl_dist_load_rms", "vl_dist_load_mdf"],
    )
    valid = df.dropna(subset=["vl_dist_load_rms", "vl_dist_load_mdf"], how="all")
    if valid.empty:
        return False

    lt1_s, lt2_s = get_lt_times(subject_id, tables["subjects"], tables["lt1"])
    lt1_min = lt1_s / 60.0 if lt1_s is not None else None
    lt2_min = lt2_s / 60.0 if lt2_s is not None else None

    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(6.5, 4.2), sharex=True)

    ax_top.plot(valid["t_min"], valid["vl_dist_load_rms"], color="#ff7f0e", linewidth=1.6, label="RMS")
    _draw_lt_lines(ax_top, lt1_min, lt2_min)
    ax_top.set_ylabel("RMS, норм. ед.")
    ax_top.legend(loc="best", frameon=False, fontsize=9)
    ax_top.grid(True, alpha=0.25)

    ax_bot.plot(valid["t_min"], valid["vl_dist_load_mdf"], color="#17becf", linewidth=1.6, label="Медианная частота")
    _draw_lt_lines(ax_bot, lt1_min, lt2_min)
    ax_bot.set_xlabel("Время, мин")
    ax_bot.set_ylabel("MDF, Гц")
    ax_bot.legend(loc="best", frameon=False, fontsize=9)
    ax_bot.grid(True, alpha=0.25)

    _save(fig, OUT_ROOT / "emg", subject_id)
    return True


# ─────────────────────── main ───────────────────────

def main() -> None:
    """Прогоняет все три типа графиков по всем участникам."""
    plt.rcParams["font.family"] = "DejaVu Sans"
    plt.rcParams["axes.unicode_minus"] = False

    tables = load_tables()
    subject_ids = sorted(tables["windows"]["subject_id"].unique().tolist())

    print(f"Участников: {len(subject_ids)}")
    print(f"Выходная папка: {OUT_ROOT}")
    print()

    stats = {"dfa": 0, "nirs": 0, "emg": 0}
    for sid in subject_ids:
        ok_dfa = plot_dfa(sid, tables)
        ok_nirs = plot_nirs(sid, tables)
        ok_emg = plot_emg(sid, tables)
        stats["dfa"] += int(ok_dfa)
        stats["nirs"] += int(ok_nirs)
        stats["emg"] += int(ok_emg)
        flags = "".join(["✓" if x else "·" for x in (ok_dfa, ok_nirs, ok_emg)])
        print(f"  {sid}  DFA/NIRS/EMG: {flags}")

    print()
    print(f"Сохранено: DFA={stats['dfa']}, NIRS={stats['nirs']}, EMG={stats['emg']}")


if __name__ == "__main__":
    main()
