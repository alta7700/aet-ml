"""Схематическая визуализация поиска LT1 и LT2 для каждого субъекта.

Использует те же функции, что и сборка датасета:
  • methods.lt2.compute_lt2          — modDmax + DFA-α1 + HHb + refined
  • build_baseline_lt1 (ниже)        — порт логики extract_lt1_labels.py

Без цифр в подписях/заголовках — только оси и легенда.
Один PNG на каждого субъекта.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import PchipInterpolator

from analysis.schemas import AnalysisConfig
from methods.lt2 import (
    LactatePoint,
    Lt2Result,
    build_lactate_points,
    compute_lt2,
    interpolate_time_for_power,
    load_fulltest,
)


# ─── LT1 (порт из scripts/extract_lt1_labels.py для self-containment) ───────

_BASELINE_DELTA_MMOL = 0.4
_BASELINE_N_POINTS = 1


@dataclass(frozen=True)
class Lt1Result:
    points: tuple[LactatePoint, ...]
    baseline_mmol: float
    threshold_mmol: float
    lt1_pchip_power_w: float
    lt1_pchip_lactate_mmol: float
    lt1_pchip_time_sec: float
    lt1_discrete_power_w: float
    lt1_discrete_lactate_mmol: float
    lt1_discrete_time_sec: float


def _build_baseline_lt1(points: tuple[LactatePoint, ...]) -> Lt1Result | None:
    """LT1 = первое пересечение PCHIP-кривой с baseline + 0.4 ммоль/л."""
    powers = np.array([p.power_w for p in points], dtype=float)
    lactates = np.array([p.lactate_mmol for p in points], dtype=float)
    if len(points) < 2:
        return None

    n_base = min(_BASELINE_N_POINTS, len(lactates))
    baseline = float(np.mean(lactates[:n_base]))
    threshold = baseline + _BASELINE_DELTA_MMOL

    above = np.where(lactates > threshold)[0]
    if len(above) == 0:
        return None
    disc_idx = int(above[0])
    lt1_disc_pow = float(powers[disc_idx])
    lt1_disc_lac = float(lactates[disc_idx])
    lt1_disc_time = float(points[disc_idx].effective_time_sec)

    pchip = PchipInterpolator(powers, lactates)
    if disc_idx == 0:
        lt1_pow = lt1_disc_pow; lt1_lac = lt1_disc_lac; lt1_time = lt1_disc_time
    else:
        lo, up = disc_idx - 1, disc_idx
        xs = np.linspace(powers[lo], powers[up], 2000)
        ys = pchip(xs)
        cross = np.where(ys >= threshold)[0]
        if len(cross) == 0:
            lt1_pow, lt1_lac, lt1_time = lt1_disc_pow, lt1_disc_lac, lt1_disc_time
        else:
            lt1_pow = float(xs[cross[0]])
            lt1_lac = float(threshold)
            lt1_time = float(interpolate_time_for_power(points, lt1_pow))

    return Lt1Result(
        points=points,
        baseline_mmol=baseline,
        threshold_mmol=threshold,
        lt1_pchip_power_w=lt1_pow,
        lt1_pchip_lactate_mmol=lt1_lac,
        lt1_pchip_time_sec=lt1_time,
        lt1_discrete_power_w=lt1_disc_pow,
        lt1_discrete_lactate_mmol=lt1_disc_lac,
        lt1_discrete_time_sec=lt1_disc_time,
    )


# ─── Поиск fulltest.h5 ─────────────────────────────────────────────────────

def _read_subject_id(h5_path: Path) -> str | None:
    """Достаёт subject_id из h5-attrs (приоритет: finaltest.h5 в той же папке)."""
    try:
        with h5py.File(h5_path, "r") as h:
            if "subject_id" not in h.attrs:
                return None
            v = h.attrs["subject_id"]
            if isinstance(v, bytes):
                return v.decode("utf-8")
            if isinstance(v, np.ndarray):
                return str(v.item())
            return str(v)
    except (OSError, KeyError):
        return None


def discover_fulltests(data_root: Path) -> dict[str, Path]:
    """Сопоставляет subject_id с fulltest.h5.

    subject_id берётся из соседнего finaltest.h5 (в fulltest.h5 его нет —
    только subject_name на русском).
    """
    out: dict[str, Path] = {}
    data_root = Path(data_root)
    for subject_dir in sorted(data_root.iterdir()):
        if not subject_dir.is_dir():
            continue
        full = subject_dir / "fulltest.h5"
        final = subject_dir / "finaltest.h5"
        if not full.exists():
            continue
        sid = _read_subject_id(final) if final.exists() else None
        if sid is None:
            continue
        out[sid] = full
    return out


# ─── Plot ──────────────────────────────────────────────────────────────────

_BLUE = "#1f77b4"
_RED = "#d62728"
_GREY = "#6c6c6c"
_HHB_COLOR = "#9467bd"
_DFA_COLOR = "#2ca02c"


def _plot_panel_lactate(ax, lt1: Lt1Result | None, lt2: Lt2Result) -> None:
    """Лактатная кривая в координатах (время теста, мин) → лактат, ммоль/л.

    Подписи на русском, тики оси X начинаются с 4 мин с шагом 3 мин
    (соответствует протоколу: 4-минутная разминка + 3-минутные ступени).
    """
    md = lt2.moddmax
    times_min = np.array([p.effective_time_sec for p in md.points], dtype=float) / 60.0
    lactates = np.array([p.lactate_mmol for p in md.points], dtype=float)

    # PCHIP по точкам (время → лактат), плотная сетка для гладкой кривой
    if len(times_min) >= 2:
        pchip = PchipInterpolator(times_min, lactates)
        fit_t = np.linspace(times_min[0], times_min[-1], 800)
        fit_l = pchip(fit_t)
    else:
        fit_t, fit_l = times_min, lactates

    # точки замеров
    ax.scatter(times_min, lactates, c="black", s=28, zorder=4,
               label="замеры лактата")
    # PCHIP кривая
    ax.plot(fit_t, fit_l, color=_GREY, lw=1.2, label="PCHIP интерполяция")

    # baseline + порог LT1
    if lt1 is not None:
        lt1_time_min = lt1.lt1_pchip_time_sec / 60.0
        ax.axhline(lt1.baseline_mmol, color=_BLUE, ls=":", lw=0.9,
                   alpha=0.6, label="базовый уровень")
        ax.axhline(lt1.threshold_mmol, color=_BLUE, ls="--", lw=1.2,
                   label="порог LT1 (базовый + 0.4)")
        ax.axvline(lt1_time_min, color=_BLUE, ls="--", lw=0.9, alpha=0.6)
        ax.scatter([lt1_time_min], [lt1.lt1_pchip_lactate_mmol],
                   marker="o", s=80, facecolor="white", edgecolor=_BLUE,
                   linewidths=2, zorder=6, label="LT1")

    # modDmax LT2: хорда + горизонталь уровня лактата + точка
    t1 = float(times_min[md.start_index]); y1 = float(lactates[md.start_index])
    t2 = float(times_min[-1]);             y2 = float(lactates[-1])
    ax.plot([t1, t2], [y1, y2], color=_RED, ls=":", lw=0.9,
            alpha=0.7, label="хорда modDmax")

    lt2_time_min = md.lt2_time_sec / 60.0
    ax.axhline(md.lt2_lactate_mmol, color=_RED, ls="--", lw=1.2,
               label="уровень лактата на LT2")
    ax.axvline(lt2_time_min, color=_RED, ls="--", lw=0.9, alpha=0.6)
    ax.scatter([lt2_time_min], [md.lt2_lactate_mmol],
               marker="s", s=80, facecolor="white", edgecolor=_RED,
               linewidths=2, zorder=6, label="LT2 (modDmax)")

    ax.set_xlabel("время теста [мин]")
    ax.set_ylabel("лактат [ммоль/л]")

    # тики: 4, 7, 10, 13, ... до конца теста
    t_max = float(times_min[-1])
    ticks = np.arange(4.0, t_max + 1.5, 3.0)
    if ticks.size:
        ax.set_xticks(ticks)
    ax.legend(fontsize=8, loc="upper left", framealpha=0.85)


def _plot_panel_power(ax, data, lt1: Lt1Result | None, lt2: Lt2Result) -> None:
    """Панель B: профиль мощности по времени с маркерами LT1/LT2."""
    t = np.asarray(data.power_times_sec, dtype=float)
    p = np.asarray(data.power_values_w, dtype=float)
    ax.step(t, p, where="post", color="black", lw=1.0, label="power profile")
    ax.axvspan(lt2.moddmax.interval_start_sec, lt2.moddmax.interval_end_sec,
               color=_RED, alpha=0.08, label="LT2 lactate interval")

    if lt1 is not None:
        ax.axvline(lt1.lt1_pchip_time_sec, color=_BLUE, ls="--", lw=1.2,
                   label="LT1 time")
    ax.axvline(lt2.moddmax.lt2_time_sec, color=_RED, ls="--", lw=1.2,
               label="LT2 time (modDmax)")
    if lt2.refined_valid:
        ax.axvline(lt2.refined_time_sec, color=_RED, ls="-", lw=1.6,
                   alpha=0.7, label="LT2 refined")

    ax.set_xlabel("time [sec]")
    ax.set_ylabel("power [W]")
    ax.legend(fontsize=7, loc="lower right", framealpha=0.85)


def _plot_panel_markers(ax, lt2: Lt2Result) -> None:
    """Панель C: DFA-α1 (левая ось) + HHb (правая ось) + кандидаты LT2."""
    dfa = lt2.dfa
    hhb = lt2.hhb

    # DFA-α1
    ax.plot(dfa.times_sec, dfa.alpha1, color=_DFA_COLOR, lw=1.0,
            label="DFA-α1")
    ax.axhline(0.5, color=_DFA_COLOR, ls="--", lw=1.0, alpha=0.6,
               label="DFA threshold")
    ax.set_xlabel("time [sec]")
    ax.set_ylabel("DFA-α1", color=_DFA_COLOR)
    ax.tick_params(axis="y", labelcolor=_DFA_COLOR)
    ax.set_ylim(0, max(1.6, float(np.nanmax(dfa.alpha1)) * 1.05
                       if dfa.alpha1.size else 1.6))

    # HHb на twin-y
    ax2 = ax.twinx()
    if hhb.smooth_times_sec.size:
        ax2.plot(hhb.smooth_times_sec, hhb.smooth_values,
                 color=_HHB_COLOR, lw=1.0, label="HHb (smoothed)")
    ax2.set_ylabel("HHb [a.u.]", color=_HHB_COLOR)
    ax2.tick_params(axis="y", labelcolor=_HHB_COLOR)

    # кандидаты LT2 как вертикали (без цифр)
    lines = []
    lines.append(ax.axvline(lt2.moddmax.lt2_time_sec, color=_RED, ls="--",
                            lw=0.9, alpha=0.7, label="modDmax candidate"))
    if dfa.crossing_time_sec is not None:
        lines.append(ax.axvline(dfa.crossing_time_sec, color=_DFA_COLOR,
                                ls="--", lw=0.9, alpha=0.7,
                                label="DFA candidate"))
    if hhb.breakpoint_time_sec is not None:
        lines.append(ax.axvline(hhb.breakpoint_time_sec, color=_HHB_COLOR,
                                ls="--", lw=0.9, alpha=0.7,
                                label="HHb candidate"))
    if lt2.refined_valid:
        lines.append(ax.axvline(lt2.refined_time_sec, color=_RED, ls="-",
                                lw=1.6, alpha=0.75, label="LT2 refined"))

    # объединённая легенда (DFA + threshold + HHb + candidates)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=7, loc="upper left", framealpha=0.85,
              ncol=2)


def make_subject_figure(subject_id: str, fulltest_path: Path,
                        out_dir: Path, dpi: int = 150) -> Path | None:
    """Строит одну фигуру для субъекта."""
    data = load_fulltest(fulltest_path)
    try:
        lt2 = compute_lt2(data)
    except (ValueError, RuntimeError) as exc:
        print(f"[skip] {subject_id}: LT2 compute failed ({exc})")
        return None

    try:
        points = build_lactate_points(data)
        lt1 = _build_baseline_lt1(points)
    except ValueError:
        lt1 = None

    fig, ax = plt.subplots(figsize=(8, 5))
    _plot_panel_lactate(ax, lt1, lt2)
    fig.suptitle(f"{subject_id}", fontsize=12)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{subject_id}.png"
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out_path


def make_all(cfg: AnalysisConfig, *, data_root: Path | None = None,
             subject: str | None = None) -> list[Path]:
    """Прогон для всех (или одного) субъектов."""
    if data_root is None:
        # из new_arch/ → ../data
        data_root = Path(__file__).resolve().parents[2] / "data"
    mapping = discover_fulltests(Path(data_root))
    if not mapping:
        raise FileNotFoundError(f"Не найден ни один fulltest.h5 в {data_root}")

    if subject is not None:
        if subject not in mapping:
            raise KeyError(f"subject {subject!r} не найден в {data_root}")
        items = [(subject, mapping[subject])]
    else:
        items = sorted(mapping.items())

    out_dir = cfg.figures_dir / "lt_search"
    saved: list[Path] = []
    for sid, path in items:
        print(f"  {sid}: ", end="", flush=True)
        p = make_subject_figure(sid, path, out_dir, dpi=cfg.figure_dpi)
        if p is not None:
            print(f"→ {p.name}")
            saved.append(p)
        else:
            print("skipped")
    return saved
