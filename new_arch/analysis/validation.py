"""Валидация training-артефактов перед сборкой analysis-кэша."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from common_lib import (
    PREDICTIONS_REQUIRED_COLUMNS, PREDICTIONS_NONNULL_COLUMNS,
    MODELS_CSV_COLUMNS,
)
from analysis.loader import DiscoveryResult


@dataclass
class ValidationReport:
    """Отчёт о валидности артефактов. ``errors`` блокируют сборку, ``warnings`` — нет."""
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    # Сюда складываются model_id, которые НЕ прошли валидацию и должны быть исключены.
    excluded_model_ids: set[str] = field(default_factory=set)

    @property
    def ok(self) -> bool:
        return not self.errors

    def to_dict(self) -> dict:
        return {
            "ok": self.ok,
            "errors": self.errors,
            "warnings": self.warnings,
            "excluded_model_ids": sorted(self.excluded_model_ids),
        }

    def save_json(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.to_dict(), indent=2, ensure_ascii=False),
            encoding="utf-8")


def validate_artifacts(disc: DiscoveryResult) -> ValidationReport:
    """Проверяет models_df и каждый predictions parquet."""
    rep = ValidationReport()

    # 1) models.csv schema
    missing_cols = [c for c in MODELS_CSV_COLUMNS if c not in disc.models_df.columns]
    if missing_cols:
        rep.errors.append(
            f"В models.csv отсутствуют колонки: {missing_cols}")

    # 2) глобальная уникальность model_id
    dups = disc.models_df["model_id"][
        disc.models_df["model_id"].duplicated(keep=False)
    ].unique().tolist()
    if dups:
        rep.errors.append(
            f"Дубликаты model_id (должны быть уникальны глобально): {dups[:5]}"
            + (" ..." if len(dups) > 5 else ""))

    # 3) missing predictions
    for mid in disc.missing_predictions:
        rep.warnings.append(f"missing predictions for model_id={mid}")
        rep.excluded_model_ids.add(mid)

    # 4) schema каждого parquet'а + NaN + invariant fold_id == loso_subject_{subject_id}
    for model_id, path in disc.predictions_paths.items():
        try:
            df = pd.read_parquet(path)
        except Exception as exc:
            rep.errors.append(f"{model_id}: не читается parquet ({exc})")
            rep.excluded_model_ids.add(model_id)
            continue

        miss = [c for c in PREDICTIONS_REQUIRED_COLUMNS if c not in df.columns]
        if miss:
            rep.errors.append(
                f"{model_id}: в parquet нет колонок {miss}")
            rep.excluded_model_ids.add(model_id)
            continue

        # NaN в критичных полях
        nan_cols = [c for c in PREDICTIONS_NONNULL_COLUMNS
                    if c in df.columns and df[c].isna().any()]
        if nan_cols:
            rep.errors.append(
                f"{model_id}: NaN в критичных колонках {nan_cols}")
            rep.excluded_model_ids.add(model_id)
            continue

        # fold_id invariant
        bad = df[df["fold_id"] != ("loso_subject_" + df["subject_id"].astype(str))]
        if not bad.empty:
            rep.errors.append(
                f"{model_id}: fold_id != loso_subject_{{subject_id}} "
                f"(пример: fold_id={bad.iloc[0]['fold_id']!r}, "
                f"subject_id={bad.iloc[0]['subject_id']!r})")
            rep.excluded_model_ids.add(model_id)
            continue

        # должны быть эпохи
        if df["epoch"].nunique() < 1:
            rep.errors.append(f"{model_id}: нет эпох в parquet")
            rep.excluded_model_ids.add(model_id)

    return rep
