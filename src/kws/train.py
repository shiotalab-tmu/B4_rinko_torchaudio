"""学習ループ（CLI）.

【第3回で穴埋めするファイル】run_epoch の学習3ステップが TODO．
完成したら確認スクリプト check_train を PASS させること．

config(YAML) を読み，学習ループを回し，checkpoint を保存する．
checkpoint は 2 種類:
  - last.pt … 毎エポックの「現在の重み」（optimizer/epoch も持つ → 中断再開に使う）
  - best.pt … val accuracy が最良だったエポックの重み（推論・配布に使う）

使い方:
    uv run python -m kws.train --config configs/baseline.yaml --device cuda
    uv run python -m kws.train --config configs/baseline.yaml --device cuda --epochs 1
    uv run python -m kws.train --config configs/baseline.yaml --device cuda --resume  # 中断再開
"""

from __future__ import annotations

import argparse
import datetime
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import yaml
from torch import nn

from kws.data import get_dataloaders
from kws.model import AudioCNN
from kws.utils import count_params, get_device, load_checkpoint, save_checkpoint, set_seed


@dataclass
class Config:
    """ハイパーパラメータ一式．YAML で上書きできる."""

    epochs: int = 15
    batch_size: int = 256
    lr: float = 1e-3
    n_mels: int = 64
    base: int = 32
    num_workers: int = 4
    seed: int = 42
    device: str | None = None
    data_root: str = "data"
    run_name: str = "baseline"
    # --- 第4回の布石 ---
    augment: bool = False
    scheduler: str | None = None  # None または "cosine"


def load_config(path: str | None, overrides: dict) -> Config:
    """YAML を読んで Config を作り，CLI の override（None でない値）を上書きする."""
    cfg = Config()
    if path:
        with open(path) as f:
            data = yaml.safe_load(f) or {}
        for key, value in data.items():
            if hasattr(cfg, key):
                setattr(cfg, key, value)
    for key, value in overrides.items():
        if value is not None:
            setattr(cfg, key, value)
    return cfg


def run_epoch(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    desc: str = "",
) -> tuple[float, float]:
    """1 epoch 分の学習 or 評価を行い，(平均 loss, accuracy) を返す.

    optimizer を渡すと学習モード，渡さないと評価モード．
    """
    is_train = optimizer is not None
    model.train(is_train)
    total, correct, loss_sum = 0, 0, 0.0

    for feats, targets in loader:
        feats, targets = feats.to(device), targets.to(device)
        with torch.set_grad_enabled(is_train):
            logits = model(feats)
            loss = criterion(logits, targets)
        if is_train:
            # TODO: 学習の3ステップを書く
            # 1. 勾配リセット
            # 2. 誤差逆伝播
            # 3. パラメータ更新
            raise NotImplementedError("run_epoch の学習ステップを実装してね")

        loss_sum += loss.item() * targets.size(0)
        correct += (logits.argmax(1) == targets).sum().item()
        total += targets.size(0)

    return loss_sum / total, correct / total


def main():
    parser = argparse.ArgumentParser(description="KWS 学習ループ")
    parser.add_argument("--config", help="YAML config のパス")
    parser.add_argument("--device", help="cuda / cpu")
    parser.add_argument("--epochs", type=int, help="config の epochs を上書き")
    parser.add_argument("--run-name", dest="run_name", help="実験名（exp/<run_name>/ に保存）")
    parser.add_argument("--resume", action="store_true", help="exp/<run_name>/last.pt から再開")
    args = parser.parse_args()

    cfg = load_config(
        args.config,
        {"device": args.device, "epochs": args.epochs, "run_name": args.run_name},
    )
    set_seed(cfg.seed)
    device = get_device(cfg.device)
    out_dir = Path("exp") / cfg.run_name
    if out_dir.exists() and not args.resume and any(out_dir.iterdir()):
        ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        backup = out_dir.with_name(f"{cfg.run_name}_backup_{ts}")
        out_dir.rename(backup)
        print(f"既存の {out_dir} を {backup} に退避した")
    out_dir.mkdir(parents=True, exist_ok=True)
    last_path, best_path = out_dir / "last.pt", out_dir / "best.pt"

    (out_dir / "config.json").write_text(
        json.dumps(asdict(cfg), indent=2, ensure_ascii=False)
    )

    loaders = get_dataloaders(cfg.data_root, cfg.batch_size, cfg.n_mels, cfg.num_workers)
    model = AudioCNN(n_classes=35, base=cfg.base).to(device)
    print(f"device: {device} / run: {cfg.run_name}")
    print(f"model params: {count_params(model):,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    scheduler = (
        torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)
        if cfg.scheduler == "cosine"
        else None
    )

    best_acc, start_epoch = 0.0, 1
    if args.resume and last_path.exists():
        ckpt = load_checkpoint(last_path, model, optimizer, map_location=device)
        start_epoch = ckpt.get("epoch", 0) + 1
        best_acc = ckpt.get("best_acc", 0.0)
        if scheduler is not None:
            for _ in range(start_epoch - 1):
                scheduler.step()
        print(f"resume: epoch {start_epoch} から再開（best_acc {best_acc:.3f}）")

    for epoch in range(start_epoch, cfg.epochs + 1):
        tr_loss, tr_acc = run_epoch(
            model, loaders["train"], criterion, device, optimizer,
        )
        va_loss, va_acc = run_epoch(
            model, loaders["val"], criterion, device, None,
        )
        if scheduler is not None:
            scheduler.step()

        print(
            f"epoch {epoch:2d}: "
            f"train_loss {tr_loss:.3f} acc {tr_acc:.3f} | "
            f"val_loss {va_loss:.3f} acc {va_acc:.3f}"
        )

        improved = va_acc > best_acc
        if improved:
            best_acc = va_acc
        save_checkpoint(
            last_path, model, optimizer, epoch, best_acc, extra={"config": asdict(cfg)}
        )
        if improved:
            save_checkpoint(
                best_path, model, optimizer, epoch, best_acc, extra={"config": asdict(cfg)}
            )

    print(f"best val acc: {best_acc:.3f}  →  {best_path}")


if __name__ == "__main__":
    main()
