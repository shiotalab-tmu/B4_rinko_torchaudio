"""汎用ユーティリティ：seed 固定・device 取得・checkpoint 保存/読込・パラメータ数."""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch


def set_seed(seed: int = 42, deterministic: bool = True) -> None:
    """乱数シードを固定して実験を再現可能にする（Python / NumPy / PyTorch）."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device(name: str | None = None) -> torch.device:
    """使う device を返す．

    name を明示指定（"cuda" / "cpu"）したらそれを尊重する．
    name が None / "auto" のときだけ自動判定（cuda があれば cuda）.
    """
    if name is None or name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "device に cuda を指定したが CUDA が使えない．"
            "GPU マシンで実行するか --device cpu を指定して．"
        )
    return device


def count_params(model: torch.nn.Module) -> int:
    """学習対象パラメータ数を数える."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    epoch: int = 0,
    best_acc: float = 0.0,
    extra: dict | None = None,
) -> None:
    """学習状態を保存する（model 必須，optimizer/epoch は途中再開用，extra は config など）."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ckpt = {"model": model.state_dict(), "epoch": epoch, "best_acc": best_acc}
    if optimizer is not None:
        ckpt["optimizer"] = optimizer.state_dict()
    if extra:
        ckpt.update(extra)
    torch.save(ckpt, path)


def load_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    map_location: str | torch.device = "cpu",
) -> dict:
    """checkpoint を読み込んで model（と optimizer）に反映し，dict 全体を返す."""
    # 自前で保存した checkpoint（config dict 等を含む）なので weights_only=False を明示．
    ckpt = torch.load(path, map_location=map_location, weights_only=False)
    model.load_state_dict(ckpt["model"])
    if optimizer is not None and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    return ckpt
