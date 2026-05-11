"""v0106c — Wavelet → Attention-LSTM + Monotonic Loss

Версия:    v0106c
Дата:      2026-05-08

Всё вместе: CWT (v0103) + Attention-LSTM (v0104) + Monotonic penalty (v0105).

Loss = HuberLoss(y_true, y_pred)
     + λ × mean(max(0, y_pred[i+1] − y_pred[i])²)

Логика: v0106a без монотонного штрафа показал, что гибрид работает.
        Добавляем физическое ограничение: время до порога только убывает.
        Ожидаем: более гладкие предсказания → Kalman сглаживание эффективнее.
"""

from __future__ import annotations

import argparse
import time
import warnings
from pathlib import Path
import sys

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import numpy as np
import pandas as pd
import pywt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
from scipy.stats import spearmanr
from joblib import Parallel, delayed

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def _load_v0011_ref() -> dict:
    """Загружает референсные MAE v0011 из best_per_set.csv; fallback на N=14 константы."""
    csv = _ROOT / "results" / "v0011" / "best_per_set.csv"
    if not csv.exists():
        return {("lt2", "EMG+NIRS+HRV"): 1.859, ("lt1", "EMG+NIRS+HRV"): 2.277}
    ref_df = pd.read_csv(csv)
    return {(row["target"], row["feature_set"]): row["kalman_mae_min"]
            for _, row in ref_df.iterrows()}


from dataset_pipeline.common import DEFAULT_DATASET_DIR
from scripts.v0011_modality_ablation import prepare_data, get_feature_cols, kalman_smooth

OUT_DIR = _ROOT / "results" / "v0106c"


# ─── Конфиг ───────────────────────────────────────────────────────────────────

class Config:
    window_step   = 4
    seq_length    = 12
    batch_size    = 16
    num_epochs    = 100
    learning_rate = 0.001
    patience      = 15
    wavelet       = "morl"
    scales        = [1, 2, 4, 8, 16]
    feat_embed    = 16
    hidden_size   = 64
    num_layers    = 2
    num_heads     = 4
    dropout       = 0.3
    mono_weight   = 1.0    # λ монотонного штрафа
    device        = "cpu"


# ─── Dataset ──────────────────────────────────────────────────────────────────

class WaveletSeqDataset(Dataset):
    def __init__(self, X, y, seq_length, window_step, scales, wavelet):
        ind_idx = np.arange(0, len(X), window_step)
        X_ind = X[ind_idx].astype(np.float32)
        y_ind = y[ind_idx].astype(np.float32)
        self.n = len(X_ind); self.seq_length = seq_length
        n_f, n_sc = X_ind.shape[1], len(scales)

        cwt_all = np.zeros((self.n, n_f, n_sc), dtype=np.float32)
        if self.n > 1:
            for f in range(n_f):
                coeffs, _ = pywt.cwt(X_ind[:, f].astype(np.float64), scales, wavelet)
                cwt_all[:, f, :] = np.abs(coeffs).T.astype(np.float32)

        flat = cwt_all.reshape(-1, n_sc)
        mean_, std_ = flat.mean(0), flat.std(0) + 1e-8
        self.cwt = ((cwt_all - mean_) / std_).reshape(self.n, n_f * n_sc).astype(np.float32)
        self.X = X_ind; self.y = y_ind

    def __len__(self): return max(0, self.n - self.seq_length + 1)

    def __getitem__(self, idx):
        x_orig = torch.from_numpy(self.X[idx: idx + self.seq_length])
        x_cwt  = torch.from_numpy(self.cwt[idx: idx + self.seq_length])
        y = torch.tensor(self.y[idx + self.seq_length - 1], dtype=torch.float32)
        return x_orig, x_cwt, y


# ─── Attention pooling ────────────────────────────────────────────────────────

class AttentionPooling(nn.Module):
    def __init__(self, hidden_size, num_heads, dropout):
        super().__init__()
        self.attn  = nn.MultiheadAttention(hidden_size, num_heads,
                                           dropout=dropout, batch_first=True)
        self.norm  = nn.LayerNorm(hidden_size)
        self.score = nn.Linear(hidden_size, 1)

    def forward(self, x):
        attn_out, _ = self.attn(x, x, x)
        attn_out = self.norm(attn_out + x)
        scores = F.softmax(self.score(attn_out), dim=1)
        return (attn_out * scores).sum(dim=1)


# ─── Модель (идентична v0106a) ─────────────────────────────────────────────────

class WaveletAttentionLSTM(nn.Module):
    def __init__(self, n_features, n_cwt, hidden_size=64,
                 num_layers=2, num_heads=4, feat_embed=16, dropout=0.3):
        super().__init__()
        self.cwt_proj = nn.Sequential(
            nn.Linear(n_cwt, feat_embed), nn.ReLU(), nn.Dropout(dropout))
        self.lstm = nn.LSTM(
            input_size=n_features + feat_embed,
            hidden_size=hidden_size, num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0, batch_first=True)
        self.attention = AttentionPooling(hidden_size, num_heads, dropout)
        self.dropout   = nn.Dropout(dropout)
        self.fc        = nn.Linear(hidden_size, 1)

    def forward(self, x_orig, x_cwt):
        x_cwt_proj = self.cwt_proj(x_cwt)
        x_in = torch.cat([x_orig, x_cwt_proj], dim=-1)
        lstm_out, _ = self.lstm(x_in)
        return self.fc(self.dropout(self.attention(lstm_out))).squeeze(1)


# ─── Монотонный штраф ─────────────────────────────────────────────────────────

def monotonic_penalty(y_pred: torch.Tensor) -> torch.Tensor:
    if len(y_pred) < 2:
        return torch.tensor(0.0, device=y_pred.device)
    return F.relu(y_pred[1:] - y_pred[:-1]).pow(2).mean()


# ─── Обучение с монотонным штрафом ────────────────────────────────────────────

def train_model(model, train_loader, val_loader, config):
    huber     = nn.HuberLoss(delta=60.0)
    optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate,
                            weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.num_epochs, eta_min=1e-5)
    best_val, patience_counter = float("inf"), 0
    best_state = {k: v.clone() for k, v in model.state_dict().items()}

    for _ in range(config.num_epochs):
        model.train()
        for x_orig, x_cwt, y in train_loader:
            x_orig, x_cwt, y = (x_orig.to(config.device),
                                 x_cwt.to(config.device), y.to(config.device))
            optimizer.zero_grad()
            y_hat = model(x_orig, x_cwt)
            loss  = huber(y_hat, y) + config.mono_weight * monotonic_penalty(y_hat)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        scheduler.step()

        model.eval(); val_loss = 0.0
        with torch.no_grad():
            for x_orig, x_cwt, y in val_loader:
                x_orig, x_cwt, y = (x_orig.to(config.device),
                                     x_cwt.to(config.device), y.to(config.device))
                # Validation: только Huber (без штрафа)
                val_loss += huber(model(x_orig, x_cwt), y).item() * len(y)
        val_loss /= max(len(val_loader.dataset), 1)

        if val_loss < best_val:
            best_val = val_loss; patience_counter = 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= config.patience: break

    model.load_state_dict(best_state)
    return model


# ─── Один LOSO фолд ───────────────────────────────────────────────────────────

def _run_one_loso_fold(test_subject_id, df, feat_cols, target_col, config):
    train = df[df["subject_id"] != test_subject_id].sort_values(
        ["subject_id", "window_start_sec"])
    test  = df[df["subject_id"] == test_subject_id].sort_values("window_start_sec")

    imp = SimpleImputer(strategy="median"); sc = StandardScaler()
    X_tr = sc.fit_transform(imp.fit_transform(train[feat_cols].values))
    X_te = sc.transform(imp.transform(test[feat_cols].values))
    y_tr = train[target_col].values; y_te = test[target_col].values

    train_ds = WaveletSeqDataset(X_tr, y_tr, config.seq_length, config.window_step,
                                  config.scales, config.wavelet)
    test_ds  = WaveletSeqDataset(X_te, y_te, config.seq_length, config.window_step,
                                  config.scales, config.wavelet)

    if len(train_ds) < 4 or len(test_ds) == 0:
        return {"fold": test_subject_id, "error": "Недостаточно данных"}

    split = max(1, int(0.8 * len(train_ds)))
    train_sub, val_sub = torch.utils.data.random_split(
        train_ds, [split, len(train_ds) - split],
        generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(train_sub, batch_size=config.batch_size, shuffle=True)
    val_loader   = DataLoader(val_sub,   batch_size=config.batch_size, shuffle=False)
    test_loader  = DataLoader(test_ds,   batch_size=config.batch_size, shuffle=False)

    n_cwt = X_tr.shape[1] * len(config.scales)
    model = WaveletAttentionLSTM(
        n_features=X_tr.shape[1], n_cwt=n_cwt,
        hidden_size=config.hidden_size, num_layers=config.num_layers,
        num_heads=config.num_heads, feat_embed=config.feat_embed,
        dropout=config.dropout,
    ).to(config.device)

    model = train_model(model, train_loader, val_loader, config)

    model.eval(); preds = []
    with torch.no_grad():
        for x_orig, x_cwt, _ in test_loader:
            preds.append(model(x_orig.to(config.device),
                               x_cwt.to(config.device)).cpu().numpy())
    y_pred_sparse = np.concatenate(preds)

    test_ind = np.arange(0, len(y_te), config.window_step)
    pred_at  = test_ind[config.seq_length - 1: config.seq_length - 1 + len(y_pred_sparse)]
    y_pred_full = np.full(len(y_te), np.nan)
    for i, idx in enumerate(pred_at):
        if idx < len(y_pred_full): y_pred_full[idx] = y_pred_sparse[i]
    x_all = np.arange(len(y_te)); valid = ~np.isnan(y_pred_full)
    y_pred_full = (np.interp(x_all, x_all[valid], y_pred_full[valid])
                   if valid.sum() >= 2 else np.full(len(y_te), np.nanmean(y_pred_full)))

    return {"fold": test_subject_id, "y_pred": y_pred_full, "y_true": y_te}


def _loso(df, feat_cols, target_col, config, n_jobs):
    subjects = sorted(df["subject_id"].unique())
    records  = Parallel(n_jobs=n_jobs, backend="loky", verbose=1)(
        delayed(_run_one_loso_fold)(s, df, feat_cols, target_col, config)
        for s in subjects)
    all_pred, all_true = [], []
    for rec in records:
        if "error" not in rec:
            all_pred.append(rec["y_pred"]); all_true.append(rec["y_true"])
    if not all_pred: return {"error": "Нет данных"}
    y_pred = np.concatenate(all_pred); y_true = np.concatenate(all_true)
    return {"y_pred": y_pred, "y_true": y_true,
            "raw_mae_min": mean_absolute_error(y_true, y_pred) / 60.0,
            "r2": r2_score(y_true, y_pred),
            "rho": float(spearmanr(y_true, y_pred).statistic)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--target",      choices=["lt1","lt2","both"], default="both")
    p.add_argument("--feature-set", nargs="+",
                   choices=["EMG","NIRS","EMG+NIRS","EMG+NIRS+HRV"],
                   default=["EMG","NIRS","EMG+NIRS","EMG+NIRS+HRV"])
    p.add_argument("--mono-weight", type=float, default=1.0)
    p.add_argument("--n-jobs", type=int, default=-1)
    p.add_argument("--dataset", type=Path,
                   default=DEFAULT_DATASET_DIR / "merged_features_ml.parquet")
    args = p.parse_args()

    config = Config()
    config.mono_weight = args.mono_weight

    print("=" * 70)
    print("v0106c — WAVELET → ATTENTION-LSTM + MONOTONIC LOSS")
    print("=" * 70)
    print(f"wavelet={config.wavelet}, scales={config.scales}, feat_embed={config.feat_embed}")
    print(f"window_step={config.window_step}, seq_length={config.seq_length}")
    print(f"mono_weight λ={config.mono_weight}\n")

    df_raw = pd.read_parquet(args.dataset)
    sp_path = DEFAULT_DATASET_DIR / "session_params.parquet"
    session_params = pd.read_parquet(sp_path) if sp_path.exists() else pd.DataFrame()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    targets = {"lt2":"target_time_to_lt2_center_sec","lt1":"target_time_to_lt1_sec"}
    if args.target != "both":
        targets = {k: v for k, v in targets.items() if k == args.target}

    sigma_grid = [30.0, 50.0, 75.0, 150.0]
    v0011_ref  = _load_v0011_ref()
    records = []

    for tgt_name, target_col in targets.items():
        print(f"\n{'═'*70}\nТАРГЕТ: {tgt_name.upper()}\n{'═'*70}\n")
        df_prep = prepare_data(df_raw, session_params, tgt_name)
        df_tgt  = df_prep.dropna(subset=[target_col])

        for fset in args.feature_set:
            feat_cols = get_feature_cols(df_tgt, fset)
            if not feat_cols: continue
            n_subj = df_tgt["subject_id"].nunique()
            print(f"  [{fset} / {tgt_name}]  n={n_subj}, {len(feat_cols)} признаков")

            t0  = time.perf_counter()
            res = _loso(df_tgt, feat_cols, target_col, config, args.n_jobs)
            elapsed = time.perf_counter() - t0
            if "error" in res: print(f"    ❌ {res['error']}"); continue

            best_mae, best_sig, k_maes = float("inf"), sigma_grid[0], {}
            for sigma in sigma_grid:
                y_k = kalman_smooth(res["y_pred"], sigma_p=5.0, sigma_obs=sigma)
                mae = mean_absolute_error(res["y_true"], y_k) / 60.0
                k_maes[sigma] = round(mae, 4)
                if mae < best_mae: best_mae, best_sig = mae, sigma

            records.append({
                "feature_set": fset, "target": tgt_name,
                "mono_weight": config.mono_weight,
                "n_subjects": n_subj, "n_features": len(feat_cols),
                "raw_mae_min": round(res["raw_mae_min"], 4),
                "kalman_mae_min": round(best_mae, 4), "best_sigma_obs": best_sig,
                "kalman_30": k_maes.get(30.0), "kalman_50": k_maes.get(50.0),
                "kalman_75": k_maes.get(75.0), "kalman_150": k_maes.get(150.0),
                "r2": round(res["r2"], 3), "rho": round(res["rho"], 3),
                "sec": round(elapsed, 1),
            })
            print(f"    raw={res['raw_mae_min']:.3f}  "
                  f"kalman_best={best_mae:.3f} (sigma={best_sig})  ({elapsed:.1f}s)")
            print(f"    sigma grid: {k_maes}")

    df_out = pd.DataFrame(records)
    df_out.to_csv(OUT_DIR / "summary.csv", index=False)

    print("\n" + "="*70 + "\nИТОГИ:")
    for _, r in df_out.sort_values(["target","kalman_mae_min"]).iterrows():
        ref   = v0011_ref.get((r["target"], r["feature_set"]))
        delta = f"  Δ={r['kalman_mae_min']-ref:+.3f} vs v0011" if ref else ""
        print(f"  {r['target'].upper()} / {r['feature_set']:<16s}  "
              f"kalman={r['kalman_mae_min']:.3f}{delta}")
    print(f"\n✅ Готово: {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
