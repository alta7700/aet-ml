"""new_arch/analysis — research-grade analysis pipeline.

Поверх артефактов training pipeline (results/{architecture_id}/...) строит
многослойный кэш и выводит metrics, ranking, comparison-таблицы, plots
и автоматические conclusion-блоки для диссертации.

Точка входа CLI: ``python -m analysis.run_analysis``.

Слои данных:
  Layer 0   results/.../models.csv, predictions_*.parquet  (training-pipeline output)
  Layer 1   analysis_out/cache/predictions_best/             (best-epoch predictions)
  Layer 2a  analysis_out/cache/subject_metrics.parquet       (regression-fit метрики)
  Layer 2b  analysis_out/cache/lt_point_metrics.parquet      (LT-time prediction метрики)
  Layer 3   analysis_out/cache/model_summary.parquet         (агрегаты по субъектам)
  Layer 4   analysis_out/comparisons/*.parquet               (парные сравнения)
"""

from __future__ import annotations

__all__ = ["schemas", "loader", "validation", "aggregation", "metrics",
           "statistics", "ranking", "plotting", "reporting", "conclusions"]
