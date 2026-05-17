"""Явные списки ArchitectureSpec для трёх семейств.

Каждый список фиксирует базовые архитектуры; конкретный model_id формируется
из ArchitectureSpec + (target, feature_set, with_abs, wavelet_mode).

build_estimator(arch) — фабрика sklearn-эстиматора для Lin*-архитектур.
"""

from __future__ import annotations

import itertools
from typing import Any

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import ElasticNet, HuberRegressor, Ridge
from sklearn.svm import SVR

from new_arch.common_lib import ArchitectureSpec


# ─── Linear / classical зоопарк (зеркало scripts/v0011_modality_ablation._build_zoo) ──

def _lin_specs() -> list[ArchitectureSpec]:
    """29 классических моделей: Ridge, Huber, ElasticNet, GBM, SVR."""
    base = dict(
        family="Lin",
        window_size_sec=30,
        sequence_length=1,
        stride_sec=0,
        sample_stride_sec=5,
        forced_wavelet_mode=None,
    )
    out: list[ArchitectureSpec] = []
    n = 0

    def add(model_class: str, hp: dict[str, Any], full: str, short: str) -> None:
        nonlocal n
        n += 1
        out.append(ArchitectureSpec(
            architecture_id=f"Lin{n}",
            architecture_name=full,
            short_architecture_name=short,
            model_class_name=model_class,
            hyperparams=hp,
            **base,
        ))

    for alpha in [1, 10, 100, 1000]:
        add("Ridge", {"alpha": alpha},
            f"Ridge regression (alpha={alpha})", f"Ridge α={alpha}")

    for eps in [1.1, 1.35, 1.5, 2.0]:
        add("HuberRegressor", {"epsilon": eps, "max_iter": 2000},
            f"Huber regression (epsilon={eps})", f"Huber ε={eps}")

    for alpha, l1 in itertools.product([0.01, 0.1, 1.0], [0.2, 0.5, 0.9]):
        add("ElasticNet",
            {"alpha": alpha, "l1_ratio": l1, "max_iter": 5000, "random_state": 42},
            f"ElasticNet (alpha={alpha}, l1_ratio={l1})",
            f"EN α={alpha}/l1={l1}")

    for n_est, depth in itertools.product([50, 100, 200], [2, 3]):
        add("GradientBoostingRegressor",
            {"n_estimators": n_est, "max_depth": depth, "random_state": 42},
            f"Gradient Boosting (n={n_est}, depth={depth})",
            f"GBM n={n_est}/d={depth}")

    for C, eps in itertools.product([1, 10, 100], [0.1, 1.0]):
        add("SVR",
            {"kernel": "rbf", "C": C, "epsilon": eps},
            f"SVR rbf (C={C}, epsilon={eps})",
            f"SVR C={C}/ε={eps}")

    return out


LINEAR_ARCHS: list[ArchitectureSpec] = _lin_specs()

# ─── LSTM семейство ─────────────────────────────────────────────────────────

# База: окно 30 сек, шаг 5 сек.
_BASE_STEP_SEC = 5


def _lstm_specs() -> list[ArchitectureSpec]:
    """LSTM-архитектуры (stateless, без val-split, без CWT в LSTM1).

    LSTM1 = stateless 6×30 (3 мин контекста, internal_stride=30 сек).
    """
    specs: list[ArchitectureSpec] = []

    common_hp = {
        "hidden_size": 64,
        "num_layers": 2,
        "dropout": 0.3,
        "lr": 1e-3,
        "weight_decay": 1e-5,
        "batch_size": 64,
        "max_epochs": 80,
        "checkpoint_every_epochs": 4,   # 5% от max_epochs=80
    }

    # LSTM1: seq_len=6, internal_stride=30 сек, outer_stride=5 сек, без CWT.
    specs.append(ArchitectureSpec(
        architecture_id="LSTM1",
        family="LSTM",
        architecture_name="Stateless LSTM, 6 × 30 сек, outer-шаг 5 сек",
        short_architecture_name="SLSTM 6x30 s5",
        model_class_name="LSTMRegressor",
        window_size_sec=30,
        sequence_length=6,
        stride_sec=30,         # internal stride (между подокнами в одном sample)
        sample_stride_sec=5,   # outer stride (между sample'ами)
        forced_wavelet_mode=None,
        hyperparams={
            "seq_len": 6,
            "internal_stride_sec": 30,
            "outer_stride_sec": 5,
            **common_hp,
        },
    ))

    return specs


LSTM_ARCHS: list[ArchitectureSpec] = _lstm_specs()


# ─── TCN семейство ──────────────────────────────────────────────────────────

def _tcn_specs() -> list[ArchitectureSpec]:
    """TCN-архитектуры (stateless, без val-split).

    TCN1 = PureTCN, seq_len=30 строк (150 сек), dilations=[1,2,4,8], RF=31.
    """
    specs: list[ArchitectureSpec] = []

    common_hp = {
        "n_channels": 32,
        "kernel_size": 3,
        "dropout": 0.3,
        "lr": 1e-3,
        "weight_decay": 1e-3,
        "batch_size": 64,
        "max_epochs": 80,
        "checkpoint_every_epochs": 4,   # 5% от max_epochs=80
    }

    # TCN1: pure dilated TCN.
    specs.append(ArchitectureSpec(
        architecture_id="TCN1",
        family="TCN",
        architecture_name="Pure dilated TCN, seq=30 строк (150 сек), RF=31",
        short_architecture_name="PureTCN 30",
        model_class_name="PureTCN",
        window_size_sec=30,
        sequence_length=30,
        stride_sec=5,          # internal stride = 1 строка (5 сек)
        sample_stride_sec=5,   # outer stride = 1 строка (5 сек)
        forced_wavelet_mode=None,
        hyperparams={
            "seq_len": 30,
            "internal_stride_sec": 5,
            "outer_stride_sec": 5,
            "dilations": [1, 2, 4, 8],
            **common_hp,
        },
    ))

    return specs


TCN_ARCHS: list[ArchitectureSpec] = _tcn_specs()


# ─── Реестр и фабрика эстиматоров ───────────────────────────────────────────

_SKLEARN_CLASSES: dict[str, type] = {
    "Ridge": Ridge,
    "HuberRegressor": HuberRegressor,
    "ElasticNet": ElasticNet,
    "GradientBoostingRegressor": GradientBoostingRegressor,
    "SVR": SVR,
}


def get_architecture(architecture_id: str) -> ArchitectureSpec:
    """Возвращает ArchitectureSpec по architecture_id (поиск во всех семействах)."""
    for spec in LINEAR_ARCHS + LSTM_ARCHS + TCN_ARCHS:
        if spec.architecture_id == architecture_id:
            return spec
    raise KeyError(
        f"Неизвестная architecture_id={architecture_id!r}. "
        f"Доступны Lin*: {[s.architecture_id for s in LINEAR_ARCHS]}"
    )


def build_estimator(arch: ArchitectureSpec):
    """Создаёт sklearn-эстиматор по ArchitectureSpec.

    Поддерживает только семейство Lin (классические модели sklearn).
    """
    if arch.family != "Lin":
        raise ValueError(
            f"build_estimator пока поддерживает только family='Lin', "
            f"получено {arch.family!r}"
        )
    cls = _SKLEARN_CLASSES.get(arch.model_class_name)
    if cls is None:
        raise KeyError(
            f"Неизвестный model_class_name={arch.model_class_name!r}. "
            f"Доступны: {list(_SKLEARN_CLASSES)}"
        )
    return cls(**arch.hyperparams)
