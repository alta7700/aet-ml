"""v0400_lib — общая библиотека для серии v0401..v0408 (TCN-grid).

Архитектуры (поле arch в TcnExperimentCfg):
  pure     — dilated TCN, WeightNorm, SpatialDropout, GAP.
             seq=30 строк (150 сек), dilations=[1,2,4,8], kernel=3, RF=31.
  medium   — расширенный контекст: seq=60 строк (5 мин), dilations=[1,2,4,8,16],
             kernel=3, RF=63.
  dwt      — Haar-DWT вдоль временной оси → две лёгкие TCN-ветви
             (approx + detail), seq=30, RF=15 на ветвь.
  wavenet  — WaveNet-стиль: kernel=2, gated activation (tanh·σ), skip-conn,
             seq=30, dilations=[1,2,4,8,16].

Общие принципы:
  • stateless sliding window, outer_stride=1 строка (5 сек).
  • WeightNorm вместо BatchNorm (нет утечки BN-статистики в LOSO).
  • SpatialDropout1d — дропаут по временному измерению (каналами).
  • weight_decay=1e-3, gradient_clip=1.0.
  • <50K параметров.
  • Вывод: те же артефакты, что v0300_lib (summary.csv, per_subject.csv, NPY).

Запуск:
    PYTHONPATH=. uv run python scripts/v0401.py --target both
"""

from __future__ import annotations

import argparse
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Optional
import sys

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
from scipy.stats import spearmanr

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from dataset_pipeline.common import DEFAULT_DATASET_DIR
from scripts.v0011_modality_ablation import (
    prepare_data, get_feature_cols, kalman_smooth,
)
# Переиспользуем датасет и список исключаемых признаков из v0300_lib
from scripts.v0300_lib import StatelessSeqDataset, EXCLUDE_ABS, _prepare_X

# ─── Конфиг ───────────────────────────────────────────────────────────────────

@dataclass
class TcnExperimentCfg:
    """Конфигурация одного TCN-эксперимента.

    name      — имя папки в results/.
    arch      — pure | medium | dwt | wavenet.
    seq_len   — число строк в скользящем окне (1 строка = 5 сек).
    with_abs  — True → включать абсолютные признаки NIRS/HRV.
    n_channels — базовая ширина каналов TCN.
    """
    name: str
    arch: Literal["pure", "medium", "dwt", "wavenet"]
    seq_len: int
    with_abs: bool = False
    n_channels: int = 32
    kernel_size: int = 3      # для pure/medium/dwt; wavenet фиксирует kernel=2
    dropout: float = 0.3
    lr: float = 1e-3
    weight_decay: float = 1e-3
    max_epochs: int = 80
    patience: int = 15
    batch_size: int = 64
    val_frac: float = 0.2
    outer_stride_rows: int = 1   # шаг между sample'ами (1 строка = 5 сек)
    target: str = "both"

    @property
    def dilations(self) -> list[int]:
        """Список коэффициентов расширения для TCN."""
        if self.arch == "pure":
            return [1, 2, 4, 8]         # RF=31, покрывает seq=30
        elif self.arch == "medium":
            return [1, 2, 4, 8, 16]     # RF=63, покрывает seq=60
        elif self.arch == "dwt":
            return [1, 2, 4]            # RF=15, на ветвь (длина после DWT ≈15)
        else:  # wavenet
            return [1, 2, 4, 8, 16]     # RF=32, покрывает seq=30


# ─── Вспомогательные слои ─────────────────────────────────────────────────────

class SpatialDropout1d(nn.Module):
    """Dropout по временному каналу: обнуляет целые каналы (как Dropout2d).

    Вход: (B, C, T). Дропаут применяется по оси T — одинаково для всего канала.
    """
    def __init__(self, p: float):
        super().__init__()
        self.p = p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.p == 0.0:
            return x
        # (B, C, T) → добавляем фиктивное пространственное измерение → Dropout2d
        return F.dropout2d(x.unsqueeze(-1), p=self.p,
                           training=True).squeeze(-1)


class TemporalBlock(nn.Module):
    """Residual-блок TCN: два WeightNorm-conv1d с дилатацией, same-padding.

    Входная проекция (downsampler) добавляется, если in_ch != out_ch.
    """
    def __init__(self, in_ch: int, out_ch: int,
                 kernel_size: int, dilation: int, dropout: float):
        super().__init__()
        # same-padding: для dilated conv без потери длины
        pad = (kernel_size - 1) * dilation // 2
        self.conv1 = nn.utils.weight_norm(
            nn.Conv1d(in_ch, out_ch, kernel_size,
                      dilation=dilation, padding=pad))
        self.conv2 = nn.utils.weight_norm(
            nn.Conv1d(out_ch, out_ch, kernel_size,
                      dilation=dilation, padding=pad))
        self.drop = SpatialDropout1d(dropout)
        self.act  = nn.ReLU()
        self.downsample = (
            nn.utils.weight_norm(nn.Conv1d(in_ch, out_ch, 1))
            if in_ch != out_ch else None
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = x if self.downsample is None else self.downsample(x)
        h = self.act(self.drop(self.conv1(x)))
        h = self.act(self.drop(self.conv2(h)))
        return self.act(h + res)


# ─── Архитектуры ──────────────────────────────────────────────────────────────

class PureTCN(nn.Module):
    """Чистый dilated TCN: стек TemporalBlock → GlobalAvgPool → Linear.

    Входные данные: (B, seq_len, F) → permute → (B, F, seq_len).
    """
    def __init__(self, input_size: int, n_channels: int,
                 kernel_size: int, dilations: list[int], dropout: float):
        super().__init__()
        layers = []
        in_ch = input_size
        for d in dilations:
            layers.append(TemporalBlock(in_ch, n_channels, kernel_size, d, dropout))
            in_ch = n_channels
        self.net  = nn.Sequential(*layers)
        self.head = nn.Linear(n_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.net(x.permute(0, 2, 1))   # (B, n_ch, T)
        h = h.mean(dim=-1)                  # GAP → (B, n_ch)
        return self.head(h).squeeze(-1)     # (B,)


class HaarDWT(nn.Module):
    """Однократное Haar-DWT вдоль временной оси.

    Фиксированные (незаучиваемые) depthwise conv1d с шагом 2.
    Вход: (B, F, T). Выход: cA, cD — каждый (B, F, T//2).
    """
    def __init__(self, n_features: int):
        super().__init__()
        # Фильтры нормализованы: [1,1]/√2 и [-1,1]/√2
        inv_sqrt2 = float(2 ** -0.5)
        w_lo = torch.full((n_features, 1, 2), inv_sqrt2)
        w_hi = torch.tensor([-inv_sqrt2, inv_sqrt2]).view(1, 1, 2).expand(
            n_features, 1, 2).clone()
        self.register_buffer("w_lo", w_lo)
        self.register_buffer("w_hi", w_hi)
        self.n_features = n_features

    def forward(self, x: torch.Tensor):
        cA = F.conv1d(x, self.w_lo, stride=2, padding=0, groups=self.n_features)
        cD = F.conv1d(x, self.w_hi, stride=2, padding=0, groups=self.n_features)
        return cA, cD


class DwtTCN(nn.Module):
    """DWT-TCN: Haar-DWT → две параллельные TCN-ветви → concat → head.

    Ветвь approx обрабатывает медленные тренды (cA),
    ветвь detail — быстрые изменения (cD).
    """
    def __init__(self, input_size: int, n_channels: int,
                 kernel_size: int, dilations: list[int], dropout: float):
        super().__init__()
        self.dwt = HaarDWT(input_size)
        # Каждая ветвь — лёгкий TCN (половина каналов)
        branch_ch = max(n_channels // 2, 8)
        def _make_branch():
            layers = []
            in_ch = input_size
            for d in dilations:
                layers.append(TemporalBlock(in_ch, branch_ch, kernel_size, d, dropout))
                in_ch = branch_ch
            return nn.Sequential(*layers)
        self.branch_a = _make_branch()
        self.branch_d = _make_branch()
        self.head = nn.Linear(branch_ch * 2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xt = x.permute(0, 2, 1)           # (B, F, T)
        cA, cD = self.dwt(xt)              # каждый (B, F, T//2)
        hA = self.branch_a(cA).mean(-1)    # GAP → (B, branch_ch)
        hD = self.branch_d(cD).mean(-1)
        return self.head(torch.cat([hA, hD], dim=-1)).squeeze(-1)


class WaveNetBlock(nn.Module):
    """WaveNet-блок: gated activation (tanh·σ) + residual + skip-conn.

    Использует kernel=2 и дилатацию. Нет строгой причинности.
    """
    def __init__(self, residual_ch: int, skip_ch: int,
                 dilation: int, dropout: float):
        super().__init__()
        # Padding 'same' для dilated conv с kernel=2
        pad = dilation   # (kernel-1)*dilation = dilation для kernel=2
        self.conv = nn.utils.weight_norm(
            nn.Conv1d(residual_ch, 2 * residual_ch, 2,
                      dilation=dilation, padding=pad))
        self.skip_conv = nn.utils.weight_norm(
            nn.Conv1d(residual_ch, skip_ch, 1))
        self.res_conv  = nn.utils.weight_norm(
            nn.Conv1d(residual_ch, residual_ch, 1))
        self.drop = SpatialDropout1d(dropout)

    def forward(self, x: torch.Tensor, skip_acc: Optional[torch.Tensor]):
        # Gated activation
        h = self.conv(x)[..., :x.shape[-1]]   # обрезаем до исходной длины
        h = self.drop(h)
        R = h.shape[1] // 2
        h = torch.tanh(h[:, :R, :]) * torch.sigmoid(h[:, R:, :])
        skip = self.skip_conv(h)
        skip_acc = skip if skip_acc is None else skip_acc + skip
        return x + self.res_conv(h), skip_acc


class WaveNetTCN(nn.Module):
    """WaveNet-style TCN: стек WaveNetBlock → ReLU → conv → ReLU → conv → скаляр."""
    def __init__(self, input_size: int, residual_ch: int, skip_ch: int,
                 dilations: list[int], dropout: float):
        super().__init__()
        self.input_proj = nn.utils.weight_norm(
            nn.Conv1d(input_size, residual_ch, 1))
        self.blocks = nn.ModuleList([
            WaveNetBlock(residual_ch, skip_ch, d, dropout) for d in dilations
        ])
        self.post = nn.Sequential(
            nn.ReLU(),
            nn.utils.weight_norm(nn.Conv1d(skip_ch, skip_ch // 2, 1)),
            nn.ReLU(),
            nn.utils.weight_norm(nn.Conv1d(skip_ch // 2, 1, 1)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.input_proj(x.permute(0, 2, 1))   # (B, R, T)
        skip_acc = None
        for blk in self.blocks:
            h, skip_acc = blk(h, skip_acc)
        out = self.post(skip_acc)   # (B, 1, T)
        return out.mean(dim=-1).squeeze(-1)   # GAP → (B,)


def build_tcn_model(cfg: TcnExperimentCfg, input_size: int) -> nn.Module:
    """Фабрика моделей по конфигу."""
    if cfg.arch in ("pure", "medium"):
        return PureTCN(input_size, cfg.n_channels,
                       cfg.kernel_size, cfg.dilations, cfg.dropout)
    elif cfg.arch == "dwt":
        return DwtTCN(input_size, cfg.n_channels,
                      cfg.kernel_size, cfg.dilations, cfg.dropout)
    else:  # wavenet
        # residual=n_channels, skip=2*n_channels
        return WaveNetTCN(input_size, cfg.n_channels,
                          cfg.n_channels * 2, cfg.dilations, cfg.dropout)


def _count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ─── Тренировка ───────────────────────────────────────────────────────────────

def _train_tcn(model: nn.Module, train_loader: DataLoader,
               val_loader: DataLoader, cfg: TcnExperimentCfg,
               device: str) -> nn.Module:
    criterion = nn.MSELoss()
    opt = optim.AdamW(model.parameters(), lr=cfg.lr,
                      weight_decay=cfg.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=cfg.max_epochs, eta_min=cfg.lr * 0.1)
    best = float("inf"); patience = 0
    best_state = {k: v.clone() for k, v in model.state_dict().items()}

    for _ in range(cfg.max_epochs):
        model.train()
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            opt.zero_grad()
            pred = model(X)
            loss = criterion(pred, y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        scheduler.step()

        model.eval(); vl = 0.0; n = 0
        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(device), y.to(device)
                pred = model(X)
                vl += criterion(pred, y).item() * len(y)
                n  += len(y)
        vl /= max(n, 1)
        if vl < best:
            best = vl; patience = 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            patience += 1
            if patience >= cfg.patience:
                break

    model.load_state_dict(best_state)
    return model


def _predict_tcn(model: nn.Module, loader: DataLoader,
                 device: str) -> np.ndarray:
    model.eval(); out = []
    with torch.no_grad():
        for X, _ in loader:
            out.append(model(X.to(device)).cpu().numpy())
    return np.concatenate(out) if out else np.array([])


# ─── LOSO-фолд ────────────────────────────────────────────────────────────────

def _device() -> str:
    if torch.cuda.is_available():  return "cuda"
    if torch.backends.mps.is_available(): return "mps"
    return "cpu"


def _loso_fold_tcn(test_s, df, feat_cols, target_col, cfg, device):
    """Один LOSO-фолд для TCN: обучаем, предсказываем на test-субъекте."""
    train_df = df[df["subject_id"] != test_s].sort_values(
        ["subject_id", "window_start_sec"])
    test_df  = df[df["subject_id"] == test_s].sort_values("window_start_sec")

    if len(test_df) < cfg.seq_len:
        return None

    X_tr, X_te = _prepare_X(df, feat_cols, None, train_df.index, test_df.index)
    y_tr_raw = train_df[target_col].values.astype(np.float32)
    y_te_raw = test_df[target_col].values.astype(np.float32)

    y_sc = StandardScaler()
    y_tr = y_sc.fit_transform(y_tr_raw.reshape(-1, 1)).ravel().astype(np.float32)
    y_te = y_sc.transform(y_te_raw.reshape(-1, 1)).ravel().astype(np.float32)

    # Строим датасеты по каждому train-субъекту отдельно (нет пересечения границ)
    train_groups = train_df.groupby("subject_id", sort=False).indices
    train_datasets, val_datasets = [], []
    for sid, idx in train_groups.items():
        idx = np.asarray(idx)
        n   = len(idx)
        cut = max(1, int(n * (1 - cfg.val_frac)))
        Xs_tr = X_tr[idx[:cut]]; ys_tr = y_tr[idx[:cut]]
        Xs_vl = X_tr[idx[cut:]]; ys_vl = y_tr[idx[cut:]]
        if len(Xs_tr) >= cfg.seq_len:
            train_datasets.append(StatelessSeqDataset(
                Xs_tr, ys_tr, cfg.seq_len,
                internal_stride_rows=1,
                outer_stride_rows=cfg.outer_stride_rows))
        if len(Xs_vl) >= cfg.seq_len:
            val_datasets.append(StatelessSeqDataset(
                Xs_vl, ys_vl, cfg.seq_len,
                internal_stride_rows=1,
                outer_stride_rows=cfg.outer_stride_rows))

    if not train_datasets or not val_datasets:
        return None

    train_ds = torch.utils.data.ConcatDataset(train_datasets)
    val_ds   = torch.utils.data.ConcatDataset(val_datasets)
    test_ds  = StatelessSeqDataset(X_te, y_te, cfg.seq_len,
                                   internal_stride_rows=1,
                                   outer_stride_rows=cfg.outer_stride_rows)
    if len(test_ds) == 0:
        return None

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=cfg.batch_size, shuffle=False)
    test_loader  = DataLoader(test_ds,  batch_size=cfg.batch_size, shuffle=False)

    model = build_tcn_model(cfg, X_tr.shape[1]).to(device)
    model = _train_tcn(model, train_loader, val_loader, cfg, device)

    pred_norm = _predict_tcn(model, test_loader, device)
    if len(pred_norm) == 0:
        return None
    pred   = y_sc.inverse_transform(pred_norm.reshape(-1, 1)).ravel()
    end_pos = test_ds.starts + (cfg.seq_len - 1)  # internal_stride=1
    y_true  = y_te_raw[end_pos]
    return {"fold": test_s, "y_pred": pred, "y_true": y_true,
            "n_params": _count_params(model)}


def _loso_tcn(df: pd.DataFrame, feat_cols: list[str], target_col: str,
              cfg: TcnExperimentCfg, device: str) -> dict:
    subjects = sorted(df["subject_id"].unique())
    all_pred, all_true, subj_rows = [], [], []

    for s in subjects:
        rec = _loso_fold_tcn(s, df, feat_cols, target_col, cfg, device)
        if rec is None:
            print(f"    [skip subj={s}] недостаточно данных")
            continue
        all_pred.append(rec["y_pred"]); all_true.append(rec["y_true"])
        mae_s = mean_absolute_error(rec["y_true"], rec["y_pred"]) / 60.0
        r2_s  = (r2_score(rec["y_true"], rec["y_pred"])
                 if len(rec["y_true"]) > 1 else float("nan"))
        subj_rows.append({"subject_id": rec["fold"],
                          "mae_min": round(mae_s, 4), "r2": round(r2_s, 3)})
        print(f"    [subj={s}] mae={mae_s:.3f} r2={r2_s:.3f} "
              f"n_test={len(rec['y_pred'])} params={rec['n_params']}")

    if not all_pred:
        return {"error": "Нет валидных результатов"}
    y_pred = np.concatenate(all_pred); y_true = np.concatenate(all_true)
    return {
        "y_pred": y_pred, "y_true": y_true,
        "raw_mae_min": mean_absolute_error(y_true, y_pred) / 60.0,
        "r2":  r2_score(y_true, y_pred),
        "rho": float(spearmanr(y_true, y_pred).statistic),
        "per_subject": subj_rows,
    }


# ─── Точка входа ──────────────────────────────────────────────────────────────

def _parse_cli(default_target: str = "both") -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--target", choices=["lt1", "lt2", "both"],
                   default=default_target)
    p.add_argument("--dataset", type=Path,
                   default=DEFAULT_DATASET_DIR / "merged_features_ml.parquet")
    return p.parse_args()


def run_tcn_experiment(cfg: TcnExperimentCfg) -> None:
    """Запускает полный LOSO-эксперимент для всех fset × target и сохраняет артефакты."""
    args = _parse_cli(cfg.target)
    cfg.target = args.target
    device = _device()

    out_dir = _ROOT / "results" / cfg.name
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print(f"{cfg.name} — arch={cfg.arch}, seq_len={cfg.seq_len}, "
          f"with_abs={cfg.with_abs}, n_ch={cfg.n_channels}")
    print(f"dilations={cfg.dilations}, kernel={cfg.kernel_size}, "
          f"target={cfg.target}, device={device}")
    print("=" * 70)

    df_raw = pd.read_parquet(args.dataset)
    sp_path = DEFAULT_DATASET_DIR / "session_params.parquet"
    session_params = pd.read_parquet(sp_path) if sp_path.exists() else pd.DataFrame()

    targets_cfg = {
        "lt1": "target_time_to_lt1_pchip_sec",
        "lt2": "target_time_to_lt2_center_sec",
    }
    if cfg.target != "both":
        targets_cfg = {cfg.target: targets_cfg[cfg.target]}

    all_records, all_subj_records = [], []
    feature_sets = ["EMG", "EMG+NIRS", "EMG+NIRS+HRV"]
    sigma_grid   = [30.0, 50.0, 75.0, 150.0]
    variant      = "with_abs" if cfg.with_abs else "noabs"

    for fset in feature_sets:
        fset_tag = fset.replace("+", "_")
        print(f"\n{'='*70}\n── НАБОР ПРИЗНАКОВ: {fset} ──")

        for tgt_name, target_col in targets_cfg.items():
            print(f"\n── ТАРГЕТ {tgt_name.upper()} ──")
            df_prep = prepare_data(df_raw, session_params, tgt_name)
            df_tgt  = df_prep.dropna(subset=[target_col])
            feat_cols_full = get_feature_cols(df_tgt, fset)
            feat_cols = (feat_cols_full if cfg.with_abs
                         else [c for c in feat_cols_full if c not in EXCLUDE_ABS])
            n_subj = df_tgt["subject_id"].nunique()
            print(f"  n_subj={n_subj}, n_features={len(feat_cols)}, "
                  f"seq_len={cfg.seq_len}")

            t0  = time.perf_counter()
            res = _loso_tcn(df_tgt, feat_cols, target_col, cfg, device)
            elapsed = time.perf_counter() - t0

            if "error" in res:
                print(f"  ❌ {res['error']}")
                continue

            for row in res.get("per_subject", []):
                all_subj_records.append({"variant": variant, "feature_set": fset,
                                         "target": tgt_name, **row})

            np.save(out_dir / f"ypred_{tgt_name}_{fset_tag}.npy", res["y_pred"])
            np.save(out_dir / f"ytrue_{tgt_name}_{fset_tag}.npy", res["y_true"])

            best_mae = float("inf"); best_sigma = sigma_grid[0]; k_maes = {}
            for sigma in sigma_grid:
                y_k = kalman_smooth(res["y_pred"], sigma_p=5.0, sigma_obs=sigma)
                mae_k = mean_absolute_error(res["y_true"], y_k) / 60.0
                k_maes[sigma] = round(mae_k, 4)
                if mae_k < best_mae:
                    best_mae = mae_k; best_sigma = sigma

            all_records.append({
                "variant": variant, "feature_set": fset, "target": tgt_name,
                "n_subjects": n_subj, "n_features": len(feat_cols),
                "raw_mae_min":    round(res["raw_mae_min"], 4),
                "kalman_mae_min": round(best_mae, 4),
                "best_sigma_obs": best_sigma,
                "kalman_30":  k_maes.get(30.0), "kalman_50":  k_maes.get(50.0),
                "kalman_75":  k_maes.get(75.0), "kalman_150": k_maes.get(150.0),
                "r2":  round(res["r2"], 3), "rho": round(res["rho"], 3),
                "sec": round(elapsed, 1),
            })
            print(f"  raw={res['raw_mae_min']:.3f}  "
                  f"kalman_best={best_mae:.3f} (σ={best_sigma})  ({elapsed:.1f}s)")

    pd.DataFrame(all_records).to_csv(out_dir / "summary.csv", index=False)
    pd.DataFrame(all_subj_records).to_csv(out_dir / "per_subject.csv", index=False)
    print(f"\n✅ Готово: {out_dir.resolve()}")
