"""CLI запуска analysis-pipeline.

Подкоманды:
  discover        — сканирует results/ и валидирует артефакты.
  build-cache     — собирает Layer 1 (predictions_best) + Layer 2 + Layer 3.
  run-stats       — собирает Layer 4 (comparisons).
  plot            — генерирует фигуры.
  report          — экспортирует CSV/XLSX/TEX.
  conclude        — пишет conclusions.md.
  all             — discover → build-cache → run-stats → plot → report → conclude.

Запуск:
  cd new_arch && uv run python -m analysis.run_analysis all \
      --config analysis/config.toml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

# Гарантируем работу как `python -m analysis.run_analysis` из new_arch/
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from analysis import (
    aggregation, comparisons as cmp_mod, conclusions, conformal as conf_mod,
    loader, lt_search_viz, plotting, ranking as ranking_mod, reporting,
    training_dynamics as td_mod, validation,
)
from analysis.schemas import AnalysisConfig, load_config


def _resolve_paths(cfg: AnalysisConfig, args: argparse.Namespace) -> AnalysisConfig:
    """Подменяет cfg.results_root / cfg.out_root из аргументов CLI."""
    if args.results is not None:
        cfg.results_root = Path(args.results)
    if args.out is not None:
        cfg.out_root = Path(args.out)
    # абсолютизируем относительно cwd
    cfg.results_root = Path(cfg.results_root).resolve()
    cfg.out_root = Path(cfg.out_root).resolve()
    cfg.force = bool(args.force)
    return cfg


# ─── stages ────────────────────────────────────────────────────────────────

def cmd_discover(cfg: AnalysisConfig) -> tuple[loader.DiscoveryResult,
                                                validation.ValidationReport]:
    disc = loader.discover_artifacts(cfg.results_root)
    rep = validation.validate_artifacts(disc)
    rep_path = cfg.out_root / "discovery_report.json"
    rep.save_json(rep_path)
    n_models = len(disc.models_df)
    n_preds = len(disc.predictions_paths)
    print(f"discover: models={n_models}  predictions={n_preds}  "
          f"missing={len(disc.missing_predictions)}  errors={len(rep.errors)}  "
          f"warnings={len(rep.warnings)}")
    if rep.errors:
        print(f"  ! errors:")
        for e in rep.errors[:10]:
            print(f"    - {e}")
        if len(rep.errors) > 10:
            print(f"    ... +{len(rep.errors)-10} more")
    print(f"→ report: {rep_path}")
    return disc, rep


def cmd_build_cache(cfg: AnalysisConfig, *, disc=None, rep=None):
    if disc is None or rep is None:
        disc, rep = cmd_discover(cfg)
    if not rep.ok:
        print("ERROR: validation report содержит errors, фикси и перезапусти.")
        sys.exit(2)

    # Layer 1
    print(f"build-cache: Layer 1 (predictions_selected, policy={cfg.epoch_policy})…")
    selection_df = aggregation.build_predictions_selected(
        disc, cfg, exclude=rep.excluded_model_ids)
    print(f"  selection: {len(selection_df)} model_id")

    preds_sel = aggregation.read_predictions_selected(cfg)
    print(f"  predictions_selected rows: {len(preds_sel)}")

    # Layer 2
    print("build-cache: Layer 2a (subject_metrics)…")
    subj = aggregation.build_subject_metrics(preds_sel, disc, cfg)
    print(f"  subject_metrics rows: {len(subj)}")

    print("build-cache: Layer 2b (lt_point_metrics)…")
    lt = aggregation.build_lt_point_metrics(preds_sel, disc, cfg)
    print(f"  lt_point_metrics rows: {len(lt)}")

    # Training dynamics + conformal — независимы от Layer 3
    print("build-cache: training dynamics…")
    dyn_df, train_sum_df = td_mod.build_training_dynamics(disc, cfg)
    print(f"  training_dynamics rows: {len(dyn_df)}; "
          f"training_summary rows: {len(train_sum_df)}")

    print(f"build-cache: conformal (alphas={list(cfg.conformal_alphas)})…")
    conf_int_df = conf_mod.build_conformal_intervals(lt, cfg)
    conf_sum_df = conf_mod.build_conformal_summary(conf_int_df, subj, cfg)
    print(f"  conformal_intervals rows: {len(conf_int_df)}; "
          f"conformal_summary rows: {len(conf_sum_df)}")

    # Layer 3 с интеграцией conformal+stability
    print("build-cache: Layer 3 (model_summary)…")
    summary = aggregation.build_model_summary(
        subj, lt, disc, cfg, selection_df,
        conformal_summary_df=conf_sum_df,
        training_summary_df=train_sum_df)
    print(f"  model_summary rows: {len(summary)}")

    # Robustness ranking (отдельная таблица; ОТЛИЧНАЯ от primary)
    primary_alpha = (cfg.conformal_alphas[0] if cfg.conformal_alphas else 0.2)
    robustness = ranking_mod.build_robustness_ranking(
        summary, policy="median", primary_alpha=primary_alpha)
    if not robustness.empty:
        cfg.robustness_ranking_path.parent.mkdir(parents=True, exist_ok=True)
        robustness.to_parquet(cfg.robustness_ranking_path, index=False)
        print(f"  robustness_ranking rows: {len(robustness)}")

    return {
        "preds_best": preds_sel,
        "preds_selected": preds_sel,
        "subject": subj, "lt": lt,
        "summary": summary, "disc": disc,
        "training_dynamics": dyn_df,
        "training_summary": train_sum_df,
        "conformal_intervals": conf_int_df,
        "conformal_summary": conf_sum_df,
        "robustness": robustness,
    }


def cmd_run_stats(cfg: AnalysisConfig, *, cache=None):
    if cache is None:
        subj = pd.read_parquet(cfg.subject_metrics_path)
        lt = pd.read_parquet(cfg.lt_point_metrics_path)
    else:
        subj, lt = cache["subject"], cache["lt"]
    print("run-stats: Layer 4 (comparisons)…")
    written = cmp_mod.build_all_comparisons(lt, subj, cfg)
    for name, p in written.items():
        print(f"  {name} → {p.relative_to(cfg.out_root)}")
    return written


def cmd_plot(cfg: AnalysisConfig, *, cache=None,
             comparisons_written=None):
    if cache is None:
        preds_sel = aggregation.read_predictions_selected(cfg)
        subj = pd.read_parquet(cfg.subject_metrics_path)
        lt = pd.read_parquet(cfg.lt_point_metrics_path)
        summary = pd.read_parquet(cfg.model_summary_path)
        conf_sum = (pd.read_parquet(cfg.conformal_summary_path)
                    if cfg.conformal_summary_path.exists() else None)
        train_dyn = (pd.read_parquet(cfg.training_dynamics_path)
                     if cfg.training_dynamics_path.exists() else None)
    else:
        preds_sel, subj, lt, summary = (
            cache["preds_selected"], cache["subject"],
            cache["lt"], cache["summary"])
        conf_sum = cache.get("conformal_summary")
        train_dyn = cache.get("training_dynamics")

    if comparisons_written is None:
        comparisons_written = {
            p.stem: p for p in cfg.comparisons_dir.glob("*.parquet")
        }
    comparison_dfs = {
        name: pd.read_parquet(path)
        for name, path in comparisons_written.items()
    }

    print("plot: фигуры…")
    out = plotting.plot_all(
        subject_metrics=subj, lt_point_metrics=lt,
        model_summary=summary, preds_best=preds_sel,
        comparisons=comparison_dfs, cfg=cfg,
        conformal_summary=conf_sum, training_dynamics=train_dyn,
    )
    total = sum(len(v) for v in out.values())
    print(f"  сохранено {total} файлов фигур в {cfg.figures_dir}")
    return out


def cmd_report(cfg: AnalysisConfig, *, cache=None,
               comparisons_written=None):
    if cache is None:
        summary = pd.read_parquet(cfg.model_summary_path)
        conf_sum = (pd.read_parquet(cfg.conformal_summary_path)
                    if cfg.conformal_summary_path.exists() else None)
        train_sum = (pd.read_parquet(cfg.training_summary_path)
                     if cfg.training_summary_path.exists() else None)
        rob = (pd.read_parquet(cfg.robustness_ranking_path)
               if cfg.robustness_ranking_path.exists() else None)
    else:
        summary = cache["summary"]
        conf_sum = cache.get("conformal_summary")
        train_sum = cache.get("training_summary")
        rob = cache.get("robustness")
    if comparisons_written is None:
        comparisons_written = {
            p.stem: p for p in cfg.comparisons_dir.glob("*.parquet")
        }
    comparison_dfs = {
        name: pd.read_parquet(path)
        for name, path in comparisons_written.items()
    }
    print("report: CSV/XLSX/TEX…")
    out = reporting.export_dissertation_tables(
        summary, comparison_dfs, cfg,
        conformal_summary=conf_sum, training_summary=train_sum,
        robustness=rob)
    print(f"  таблиц сохранено: {len(out)}, workbook={out.get('workbook')}")
    return out


def cmd_conclude(cfg: AnalysisConfig, *, cache=None,
                 comparisons_written=None):
    if cache is None:
        summary = pd.read_parquet(cfg.model_summary_path)
        conf_sum = (pd.read_parquet(cfg.conformal_summary_path)
                    if cfg.conformal_summary_path.exists() else None)
        train_sum = (pd.read_parquet(cfg.training_summary_path)
                     if cfg.training_summary_path.exists() else None)
    else:
        summary = cache["summary"]
        conf_sum = cache.get("conformal_summary")
        train_sum = cache.get("training_summary")
    if comparisons_written is None:
        comparisons_written = {
            p.stem: p for p in cfg.comparisons_dir.glob("*.parquet")
        }
    comparison_dfs = {
        name: pd.read_parquet(path)
        for name, path in comparisons_written.items()
    }
    out = conclusions.build_conclusions(
        summary, comparison_dfs, cfg,
        conformal_summary=conf_sum, training_summary=train_sum)
    print(f"conclude: {out}")
    return out


# ─── argparse ──────────────────────────────────────────────────────────────

def _add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--config", type=Path, default=Path("analysis/config.toml"))
    p.add_argument("--results", type=Path, default=None,
                   help="override paths.results_root")
    p.add_argument("--out", type=Path, default=None,
                   help="override paths.out_root")
    p.add_argument("--force", action="store_true",
                   help="игнорировать кэш и пересобрать всё")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="new_arch analysis pipeline (CLI)")
    sub = parser.add_subparsers(dest="cmd", required=True)
    for name in ("discover", "build-cache", "run-stats", "plot",
                 "report", "conclude", "all"):
        sp = sub.add_parser(name)
        _add_common(sp)

    sp = sub.add_parser("viz-lt-search",
                        help="схематические фигуры поиска LT1/LT2 по субъектам")
    _add_common(sp)
    sp.add_argument("--subject", type=str, default=None,
                    help="один subject_id; без флага — все")
    sp.add_argument("--data-root", type=Path, default=None,
                    help="путь к каталогу с подпапками субъектов (default: ../data)")

    args = parser.parse_args()
    cfg = load_config(args.config if args.config.exists() else None)
    cfg = _resolve_paths(cfg, args)

    if args.cmd == "discover":
        cmd_discover(cfg)
        return
    if args.cmd == "build-cache":
        cmd_build_cache(cfg)
        return
    if args.cmd == "run-stats":
        cmd_run_stats(cfg)
        return
    if args.cmd == "plot":
        cmd_plot(cfg)
        return
    if args.cmd == "report":
        cmd_report(cfg)
        return
    if args.cmd == "conclude":
        cmd_conclude(cfg)
        return

    if args.cmd == "viz-lt-search":
        saved = lt_search_viz.make_all(
            cfg, data_root=args.data_root, subject=args.subject)
        print(f"viz-lt-search: сохранено {len(saved)} фигур в "
              f"{cfg.figures_dir / 'lt_search'}")
        return

    # all: один сквозной прогон, передаём кэш по цепочке
    disc, rep = cmd_discover(cfg)
    cache = cmd_build_cache(cfg, disc=disc, rep=rep)
    written = cmd_run_stats(cfg, cache=cache)
    cmd_plot(cfg, cache=cache, comparisons_written=written)
    cmd_report(cfg, cache=cache, comparisons_written=written)
    cmd_conclude(cfg, cache=cache, comparisons_written=written)
    print("\n✓ analysis pipeline complete.")


if __name__ == "__main__":
    main()
