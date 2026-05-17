"""Smoke-тесты для new_arch/common_lib.py.

Запуск: PYTHONPATH=. uv run python _smoke_test.py

Покрытие:
  • детерминизм build_model_id;
  • build_fold_id для int / str-числа / строкового id;
  • валидация predictions: missing/extra/NaN;
  • upsert поведение save_models_csv;
  • forced_wavelet_mode валидация в ExperimentMetadata.from_arch;
  • save_model_checkpoint для sklearn (joblib) и для torch (если torch есть).
"""

from __future__ import annotations

import json
import tempfile
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from common_lib import (
    ArchitectureSpec, ExperimentMetadata,
    build_model_id, build_fold_id,
    build_predictions_filename, build_checkpoint_filename,
    build_grouped_checkpoint_filename,
    save_models_csv, save_model_checkpoint, save_predictions_parquet,
    save_grouped_checkpoint,
    validate_predictions_dataframe,
    PREDICTIONS_REQUIRED_COLUMNS,
)


def _make_arch(arch_id: str = "Lin1",
               forced_wavelet_mode=None,
               hyperparams=None) -> ArchitectureSpec:
    return ArchitectureSpec(
        architecture_id=arch_id,
        family="Lin",
        architecture_name="Ridge regression",
        short_architecture_name="Ridge",
        model_class_name="Ridge",
        window_size_sec=30,
        sequence_length=1,
        stride_sec=0,
        sample_stride_sec=5,
        forced_wavelet_mode=forced_wavelet_mode,
        hyperparams=hyperparams or {"alpha": 1.0},
    )


def _make_meta(arch=None, **overrides) -> ExperimentMetadata:
    arch = arch or _make_arch()
    cfg = dict(target="lt1", feature_set="EMG", with_abs=True, wavelet_mode="none")
    cfg.update(overrides)
    return ExperimentMetadata.from_arch(arch, **cfg)


def _good_predictions_df(meta: ExperimentMetadata, n: int = 5,
                         epoch: int = 0) -> pd.DataFrame:
    return pd.DataFrame({
        "model_id": [meta.model_id] * n,
        "fold_id": ["loso_subject_01"] * n,
        "subject_id": ["S001"] * n,
        "epoch": [int(epoch)] * n,
        "window_size_sec": [meta.window_size_sec] * n,
        "sequence_length": [meta.sequence_length] * n,
        "stride_sec": [meta.stride_sec] * n,
        "sample_stride_sec": [meta.sample_stride_sec] * n,
        "sample_index": list(range(n)),
        "sample_start_sec": [float(i * 5) for i in range(n)],
        "sample_end_sec": [float(i * 5 + 30) for i in range(n)],
        "y_true": np.arange(n, dtype=float) * 60.0,
        "y_pred": np.arange(n, dtype=float) * 60.0 + 1.0,
    })


def test_build_model_id_determinism():
    arch = _make_arch()
    a = build_model_id(arch, target="lt1", feature_set="EMG", with_abs=True, wavelet_mode="none")
    b = build_model_id(arch, target="lt1", feature_set="EMG", with_abs=True, wavelet_mode="none")
    assert a == b, f"model_id не детерминирован: {a} != {b}"
    assert a.startswith("Lin1_"), f"prefix: {a}"
    assert len(a.split("_")[1]) == 8, f"hash длина: {a}"

    c = build_model_id(arch, target="lt2", feature_set="EMG", with_abs=True, wavelet_mode="none")
    assert a != c, "разные target должны давать разный model_id"
    print("OK: build_model_id детерминирован, формат {arch_id}_{8 hex}")


def test_build_fold_id():
    assert build_fold_id(7) == "loso_subject_07"
    assert build_fold_id("7") == "loso_subject_07"
    assert build_fold_id("S007") == "loso_subject_S007"
    print("OK: build_fold_id поддерживает int / str-число / строковый id")


def test_validate_missing():
    meta = _make_meta()
    df = _good_predictions_df(meta).drop(columns=["y_pred"])
    try:
        validate_predictions_dataframe(df)
    except ValueError as e:
        assert "y_pred" in str(e)
        print(f"OK: missing → ValueError ({e})")
        return
    raise AssertionError("ожидался ValueError")


def test_validate_extra_warning():
    meta = _make_meta()
    df = _good_predictions_df(meta).assign(extra_col=1)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        validate_predictions_dataframe(df)
    assert any("extra_col" in str(x.message) for x in w), "ожидался warning"
    print("OK: extra columns → warning, не ошибка")


def test_validate_nan_critical():
    meta = _make_meta()
    df = _good_predictions_df(meta)
    df.loc[0, "sample_index"] = np.nan
    try:
        validate_predictions_dataframe(df)
    except ValueError as e:
        assert "sample_index" in str(e)
        print(f"OK: NaN в критической колонке → ValueError")
        return
    raise AssertionError("ожидался ValueError")


def test_save_grouped_checkpoint():
    try:
        import torch
        import torch.nn as nn
    except ImportError:
        print("SKIP: torch недоступен")
        return
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        meta = _make_meta()
        md = root / meta.architecture_id / meta.model_id
        states = {
            "loso_subject_S001": nn.Linear(3, 1).state_dict(),
            "loso_subject_S002": nn.Linear(3, 1).state_dict(),
        }
        path = save_grouped_checkpoint(states, md, meta, epoch=8)
        assert path.name == f"model_{meta.model_id}_epoch-008.pt"
        loaded = torch.load(path, weights_only=True, map_location="cpu")
        assert set(loaded.keys()) == set(states.keys())
        assert "weight" in loaded["loso_subject_S001"]
        print("OK: save_grouped_checkpoint → один .pt с dict[fold_id]→state_dict")


def test_save_models_csv_upsert():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        meta1 = _make_meta()
        save_models_csv(meta1, root)
        save_models_csv(meta1, root)  # повторно тот же model_id

        csv = pd.read_csv(root / "Lin1" / "models.csv")
        assert len(csv) == 1, f"upsert: ожидалась 1 строка, получено {len(csv)}"

        # Второй model_id — должна добавиться вторая строка.
        meta2 = _make_meta(target="lt2")
        save_models_csv(meta2, root)
        csv = pd.read_csv(root / "Lin1" / "models.csv")
        assert len(csv) == 2, f"вторая строка: получено {len(csv)}"
        print("OK: save_models_csv upsert по model_id")


def test_forced_wavelet_mode():
    arch = _make_arch(arch_id="TCN3", forced_wavelet_mode="dwt")
    # Совпадает — OK.
    meta = ExperimentMetadata.from_arch(
        arch, target="lt1", feature_set="EMG", with_abs=True, wavelet_mode="dwt",
    )
    assert meta.wavelet_mode == "dwt"

    try:
        ExperimentMetadata.from_arch(
            arch, target="lt1", feature_set="EMG", with_abs=True, wavelet_mode="none",
        )
    except ValueError as e:
        assert "TCN3" in str(e)
        print(f"OK: forced_wavelet_mode защищает от противоречий")
        return
    raise AssertionError("ожидался ValueError")


def test_invalid_wavelet_mode():
    arch = _make_arch()
    try:
        ExperimentMetadata.from_arch(
            arch, target="lt1", feature_set="EMG", with_abs=True, wavelet_mode="invalid",  # type: ignore[arg-type]
        )
    except ValueError as e:
        print(f"OK: невалидный wavelet_mode → ValueError")
        return
    raise AssertionError("ожидался ValueError")


def test_save_checkpoint_sklearn():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        meta = _make_meta()
        md = root / meta.architecture_id / meta.model_id
        model = Ridge(alpha=1.0).fit(np.eye(3), np.arange(3, dtype=float))
        path = save_model_checkpoint(model, md, meta, "loso_subject_01")
        assert path.suffix == ".joblib", f"sklearn → .joblib, получено {path.suffix}"
        assert path.exists()
        import joblib
        loaded = joblib.load(path)
        assert hasattr(loaded, "coef_")
        print("OK: save_model_checkpoint для sklearn → .joblib")


def test_save_checkpoint_torch():
    try:
        import torch
        import torch.nn as nn
    except ImportError:
        print("SKIP: torch недоступен — пропуск проверки .pt checkpoint")
        return
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        meta = _make_meta()
        md = root / meta.architecture_id / meta.model_id
        model = nn.Linear(3, 1)
        path = save_model_checkpoint(model, md, meta, "loso_subject_01")
        assert path.suffix == ".pt", f"torch → .pt, получено {path.suffix}"
        state = torch.load(path, weights_only=True)
        assert "weight" in state
        print("OK: save_model_checkpoint для torch → .pt (state_dict)")


def test_save_predictions_roundtrip():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        meta = _make_meta()
        md = root / meta.architecture_id / meta.model_id
        df = _good_predictions_df(meta)
        path = save_predictions_parquet(df, md, meta)
        assert path.name == f"predictions_{meta.model_id}.parquet"
        loaded = pd.read_parquet(path)
        assert set(loaded.columns) >= set(PREDICTIONS_REQUIRED_COLUMNS)
        assert len(loaded) == len(df)
        print("OK: save_predictions_parquet roundtrip")


def main() -> None:
    test_build_model_id_determinism()
    test_build_fold_id()
    test_validate_missing()
    test_validate_extra_warning()
    test_validate_nan_critical()
    test_save_grouped_checkpoint()
    test_save_models_csv_upsert()
    test_forced_wavelet_mode()
    test_invalid_wavelet_mode()
    test_save_checkpoint_sklearn()
    test_save_checkpoint_torch()
    test_save_predictions_roundtrip()
    print("\nВсе smoke-тесты пройдены ✅")


if __name__ == "__main__":
    main()
