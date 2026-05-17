"""tcn_runner — обучение одной TCN*-архитектуры с LOSO.

Без val-split, фиксированное число эпох, AdamW + CosineAnnealingLR.
Промежуточные checkpoints каждые checkpoint_every_epochs.
predictions.parquet строится по финальной модели каждого fold.

Сохраняемые артефакты:
  results/{architecture_id}/models.csv
  results/{architecture_id}/{model_id}/history.csv
  results/{architecture_id}/{model_id}/model_{model_id}_fold-{fold_id}_epoch-{NN}.pt
  results/{architecture_id}/{model_id}/predictions_{model_id}.parquet

Пример:
  PYTHONPATH=. uv run python new_arch/tcn_runner.py \
      --architecture TCN1 --target lt1 --feature-set EMG+NIRS+HRV
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from torch.utils.data import ConcatDataset, DataLoader

from new_arch.architectures import get_architecture
from new_arch.common_lib import (
    ExperimentMetadata, arch_dir, build_fold_id, model_dir,
    save_history_csv, save_model_checkpoint, save_models_csv,
    save_predictions_parquet,
)
from new_arch.dataset_pipeline.common import DEFAULT_DATASET_DIR
from new_arch.features import get_feature_cols, prepare_data
from new_arch.models.lstm import StatelessSeqDataset
from new_arch.models.tcn import PureTCN
from new_arch.training_utils import get_device, prepare_X_for_fold

RESULTS_ROOT = Path(__file__).resolve().parent / "results"
BASE_STEP_SEC = 5

TARGET_COLS = {
    "lt1": "target_time_to_lt1_pchip_sec",
    "lt2": "target_time_to_lt2_center_sec",
}

# Реестр TCN-моделей по model_class_name.
_TCN_BUILDERS = {
    "PureTCN": lambda input_size, hp: PureTCN(
        input_size=input_size,
        n_channels=int(hp["n_channels"]),
        kernel_size=int(hp["kernel_size"]),
        dilations=list(hp["dilations"]),
        dropout=float(hp["dropout"]),
    ),
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="TCN runner (LOSO без val-split)")
    p.add_argument("--architecture", required=True, help="например TCN1")
    p.add_argument("--target", choices=["lt1", "lt2"], default="lt1")
    p.add_argument("--feature-set",
                   choices=["EMG", "NIRS", "HRV", "EMG+NIRS", "EMG+NIRS+HRV"],
                   default="EMG+NIRS+HRV")
    p.add_argument("--with-abs", dest="with_abs", action="store_true", default=True)
    p.add_argument("--no-abs", dest="with_abs", action="store_false")
    p.add_argument("--wavelet-mode",
                   choices=["none", "dwt", "cwt", "wavelet_features", "wavelet_cnn"],
                   default="none")
    p.add_argument("--max-epochs", type=int, default=None)
    p.add_argument("--checkpoint-every", type=int, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--dataset", type=Path,
                   default=DEFAULT_DATASET_DIR / "merged_features_ml.parquet")
    p.add_argument("--session-params", type=Path,
                   default=DEFAULT_DATASET_DIR / "session_params.parquet")
    p.add_argument("--results-root", type=Path, default=RESULTS_ROOT)
    return p.parse_args()


def _seed_all(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _build_train_dataset(df_train: pd.DataFrame, X_tr: np.ndarray, y_tr: np.ndarray,
                        seq_len: int, internal_stride_rows: int,
                        outer_stride_rows: int) -> ConcatDataset | None:
    """Per-subject склейка StatelessSeqDataset."""
    groups = df_train.groupby("subject_id", sort=False).indices
    parts: list[StatelessSeqDataset] = []
    span = (seq_len - 1) * internal_stride_rows + 1
    for _, idx in groups.items():
        idx = np.asarray(idx)
        if len(idx) < span:
            continue
        ds = StatelessSeqDataset(
            X_tr[idx], y_tr[idx],
            seq_len, internal_stride_rows, outer_stride_rows,
        )
        if len(ds) > 0:
            parts.append(ds)
    if not parts:
        return None
    return ConcatDataset(parts)


def _train_one_fold(model: nn.Module, loader: DataLoader,
                    *, max_epochs: int, checkpoint_every: int,
                    lr: float, weight_decay: float, device: str,
                    save_ckpt) -> list[dict]:
    """LOSO fold: AdamW + CosineAnnealingLR на T_max=max_epochs.

    save_ckpt(epoch:int) — колбек сохранения чекпоинта.
    """
    criterion = nn.MSELoss()
    opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=max_epochs, eta_min=lr * 0.1,
    )

    history: list[dict] = []
    for epoch in range(1, max_epochs + 1):
        model.train()
        loss_sum = 0.0
        mae_sum = 0.0
        n = 0
        for X, y in loader:
            X = X.to(device); y = y.to(device)
            opt.zero_grad()
            pred = model(X)
            loss = criterion(pred, y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            bs = y.size(0)
            loss_sum += float(loss.item()) * bs
            mae_sum += float((pred - y).abs().mean().item()) * bs
            n += bs

        cur_lr = float(opt.param_groups[0]["lr"])
        scheduler.step()

        history.append({
            "epoch": epoch,
            "train_loss": loss_sum / max(n, 1),
            "val_loss": float("nan"),
            "train_mae": mae_sum / max(n, 1),
            "val_mae": float("nan"),
            "lr": cur_lr,
        })

        if epoch % checkpoint_every == 0 or epoch == max_epochs:
            save_ckpt(epoch)

    return history


def _predict_fold(model: nn.Module, X_te: np.ndarray, y_te: np.ndarray,
                  seq_len: int, internal_stride_rows: int,
                  outer_stride_rows: int, batch_size: int,
                  device: str) -> tuple[np.ndarray, np.ndarray]:
    ds = StatelessSeqDataset(X_te, y_te, seq_len,
                             internal_stride_rows, outer_stride_rows)
    if len(ds) == 0:
        return np.array([], dtype=np.float32), np.array([], dtype=np.int64)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
    model.eval()
    out: list[np.ndarray] = []
    with torch.no_grad():
        for X, _ in loader:
            pred = model(X.to(device))
            out.append(pred.cpu().numpy())
    y_pred_norm = np.concatenate(out) if out else np.array([], dtype=np.float32)
    end_pos = ds.starts + (seq_len - 1) * internal_stride_rows
    return y_pred_norm, end_pos


def _build_tcn_model(arch, input_size: int) -> nn.Module:
    builder = _TCN_BUILDERS.get(arch.model_class_name)
    if builder is None:
        raise KeyError(
            f"Неизвестный TCN model_class_name={arch.model_class_name!r}. "
            f"Доступны: {list(_TCN_BUILDERS)}"
        )
    return builder(input_size, arch.hyperparams)


def run(args: argparse.Namespace) -> None:
    arch = get_architecture(args.architecture)
    if arch.family != "TCN":
        raise SystemExit(
            f"tcn_runner работает только с family='TCN', "
            f"получено {arch.family!r} для {arch.architecture_id}"
        )

    hp = arch.hyperparams
    seq_len = int(hp["seq_len"])
    int_stride_sec = int(hp["internal_stride_sec"])
    out_stride_sec = int(hp["outer_stride_sec"])
    assert int_stride_sec % BASE_STEP_SEC == 0
    assert out_stride_sec % BASE_STEP_SEC == 0
    int_stride_rows = int_stride_sec // BASE_STEP_SEC
    out_stride_rows = out_stride_sec // BASE_STEP_SEC
    lr = float(hp["lr"])
    weight_decay = float(hp["weight_decay"])
    batch_size = int(hp["batch_size"])
    max_epochs = int(args.max_epochs or hp["max_epochs"])
    checkpoint_every = int(args.checkpoint_every or hp["checkpoint_every_epochs"])
    checkpoint_every = max(1, checkpoint_every)

    meta = ExperimentMetadata.from_arch(
        arch,
        target=args.target, feature_set=args.feature_set,
        with_abs=args.with_abs, wavelet_mode=args.wavelet_mode,
    )
    device = get_device()
    _seed_all(args.seed)

    print(f"[{arch.architecture_id}] {arch.architecture_name}")
    print(f"  model_id={meta.model_id}  device={device}")
    print(f"  target={meta.target}  feature_set={meta.feature_set}  "
          f"with_abs={meta.with_abs}  wavelet_mode={meta.wavelet_mode}")
    print(f"  seq_len={seq_len}  int_stride={int_stride_sec}s  "
          f"out_stride={out_stride_sec}s")
    print(f"  max_epochs={max_epochs}  checkpoint_every={checkpoint_every}")

    md = model_dir(args.results_root, meta)
    md.mkdir(parents=True, exist_ok=True)
    save_models_csv(meta, args.results_root)
    print(f"  → {arch_dir(args.results_root, arch.architecture_id) / 'models.csv'}")

    df_raw = pd.read_parquet(args.dataset)
    session_params = pd.read_parquet(args.session_params) if args.session_params.exists() else pd.DataFrame()

    df_prep = prepare_data(df_raw, session_params, meta.target)
    target_col = TARGET_COLS[meta.target]
    df_prep = df_prep.dropna(subset=[target_col])
    feat_cols = get_feature_cols(df_prep, meta.feature_set, with_abs=meta.with_abs)
    if not feat_cols:
        raise SystemExit(f"Пустой feature_set для {meta.feature_set}")

    print(f"  n_subjects={df_prep['subject_id'].nunique()}  "
          f"n_features={len(feat_cols)}  n_windows={len(df_prep)}")

    subjects = sorted(df_prep["subject_id"].unique())
    pred_rows: list[pd.DataFrame] = []
    history_rows: list[dict] = []

    t_total = time.perf_counter()
    for test_s in subjects:
        fold_id = build_fold_id(test_s)
        t0 = time.perf_counter()

        train_df = df_prep[df_prep["subject_id"] != test_s].sort_values(
            ["subject_id", "window_start_sec"]
        )
        test_df = df_prep[df_prep["subject_id"] == test_s].sort_values(
            "window_start_sec"
        ).reset_index()
        orig_test_index = pd.Index(test_df.pop("index").values)

        X_tr, X_te = prepare_X_for_fold(
            df_prep, feat_cols, train_df.index, orig_test_index, cwt=None,
        )
        y_tr_raw = train_df[target_col].values.astype(np.float32)
        y_te_raw = test_df[target_col].values.astype(np.float32)
        y_sc = StandardScaler()
        y_tr = y_sc.fit_transform(y_tr_raw.reshape(-1, 1)).ravel().astype(np.float32)

        train_ds = _build_train_dataset(
            train_df, X_tr, y_tr,
            seq_len, int_stride_rows, out_stride_rows,
        )
        if train_ds is None:
            print(f"  [skip {fold_id}] недостаточно train-данных")
            continue
        train_loader = DataLoader(
            train_ds, batch_size=batch_size, shuffle=True,
            num_workers=args.num_workers, drop_last=False,
        )

        model = _build_tcn_model(arch, input_size=X_tr.shape[1]).to(device)
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        def _save_ckpt(epoch: int, *, _model=model, _meta=meta, _md=md, _fold_id=fold_id):
            save_model_checkpoint(_model, _md, _meta, _fold_id, epoch=epoch)

        fold_history = _train_one_fold(
            model, train_loader,
            max_epochs=max_epochs, checkpoint_every=checkpoint_every,
            lr=lr, weight_decay=weight_decay, device=device,
            save_ckpt=_save_ckpt,
        )
        for row in fold_history:
            history_rows.append({
                "model_id": meta.model_id,
                "architecture_id": meta.architecture_id,
                "fold_id": fold_id,
                **row,
            })

        y_pred_norm, end_pos = _predict_fold(
            model, X_te, y_te_raw,
            seq_len, int_stride_rows, out_stride_rows, batch_size, device,
        )
        if len(y_pred_norm) == 0:
            print(f"  [skip {fold_id}] недостаточно test-окон")
            continue
        y_pred = y_sc.inverse_transform(y_pred_norm.reshape(-1, 1)).ravel()
        y_true = y_te_raw[end_pos]

        win_start = test_df["window_start_sec"].astype(float).values
        sample_start_sec = win_start[end_pos] - (seq_len - 1) * int_stride_sec
        sample_end_sec = win_start[end_pos] + float(meta.window_size_sec)

        n_pred = len(y_pred)
        fold_df = pd.DataFrame({
            "model_id": meta.model_id,
            "fold_id": fold_id,
            "subject_id": test_df["subject_id"].values[end_pos],
            "window_size_sec": meta.window_size_sec,
            "sequence_length": meta.sequence_length,
            "stride_sec": meta.stride_sec,
            "sample_stride_sec": meta.sample_stride_sec,
            "sample_index": np.arange(n_pred, dtype=np.int64),
            "sample_start_sec": sample_start_sec.astype(float),
            "sample_end_sec": sample_end_sec.astype(float),
            "y_true": y_true.astype(float),
            "y_pred": y_pred.astype(float),
        })
        pred_rows.append(fold_df)

        mae_min = float(np.mean(np.abs(y_true - y_pred))) / 60.0
        elapsed = time.perf_counter() - t0
        n_ckpts = sum(
            1 for e in range(1, max_epochs + 1)
            if e % checkpoint_every == 0 or e == max_epochs
        )
        print(f"  fold {fold_id:<22s}  n={n_pred:<4d}  MAE={mae_min:.3f} мин  "
              f"params={n_params}  ckpts={n_ckpts}  ({elapsed:.1f}s)")

    save_history_csv(history_rows, md, meta)
    preds = pd.concat(pred_rows, ignore_index=True)
    save_predictions_parquet(preds, md, meta)

    overall_mae = float(np.mean(np.abs(preds["y_true"] - preds["y_pred"]))) / 60.0
    print(f"\n  Всего {len(subjects)} folds, "
          f"{time.perf_counter() - t_total:.1f}s; "
          f"overall MAE={overall_mae:.3f} мин")
    print(f"  → {md}/history.csv  ({len(history_rows)} строк)")
    print(f"  → {md}/predictions_{meta.model_id}.parquet  ({len(preds)} строк)")


if __name__ == "__main__":
    run(parse_args())
