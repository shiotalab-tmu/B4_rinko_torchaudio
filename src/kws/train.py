"""学習ループ（CLI）.

【第3回で穴埋めするファイル】run_epoch の学習3ステップが TODO．完成したら確認スクリプト `check_train` を PASS させること（模範解答は配布しない）．

config(YAML) を読み，学習ループを回し，loss/acc を記録し，checkpoint を保存する．
checkpoint は 2 種類:
  - last.pt … 毎エポックの「現在の重み」（optimizer/epoch も持つ → 中断再開に使う）
  - best.pt … val accuracy が最良だったエポックの重み（推論・配布に使う）
ログは wandb（任意）と history.json の両方に残す．

使い方:
    uv run python -m kws.train --config configs/baseline.yaml --device cuda
    uv run python -m kws.train --config configs/baseline.yaml --device cuda --epochs 1 --no-wandb
    uv run python -m kws.train --config configs/baseline.yaml --device cuda --resume  # 中断再開
"""

from __future__ import annotations

import argparse
import datetime
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib
import torch
import yaml
from torch import nn
from tqdm import tqdm

matplotlib.use("Agg")  # 画面の無い環境でも図を保存できるように
import matplotlib.pyplot as plt  # noqa: E402

from kws.data import get_dataloaders  # noqa: E402
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
    device: str | None = None  # None なら auto，"cuda"/"cpu" で明示指定
    data_root: str = "data"
    run_name: str = "baseline"
    # --- 第4回の布石（最小フック）---
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

    optimizer を渡すと学習モード，渡さないと評価モード（勾配を流さない）．
    学習ループの肝はこの 5 行:
        optimizer.zero_grad() → forward → loss → loss.backward() → optimizer.step()
    """
    is_train = optimizer is not None
    model.train(is_train)
    total, correct, loss_sum = 0, 0, 0.0

    for feats, targets in tqdm(loader, desc=desc, leave=False):
        feats, targets = feats.to(device), targets.to(device)
        with torch.set_grad_enabled(is_train):
            logits = model(feats)              # forward
            loss = criterion(logits, targets)  # loss
        if is_train:
            # TODO(第3回): 学習の3ステップ（勾配リセット → backward → step）
            #   optimizer.zero_grad() / loss.backward() / optimizer.step()
            raise NotImplementedError("run_epoch の学習ステップを実装してね（notebook 03 参照）")

        loss_sum += loss.item() * targets.size(0)
        correct += (logits.argmax(1) == targets).sum().item()
        total += targets.size(0)

    return loss_sum / total, correct / total


def save_loss_curve(history: list[dict], path) -> None:
    """history から loss / accuracy 曲線を描いて保存する（毎エポック更新される）."""
    ep = [h["epoch"] for h in history]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.2))
    a1.plot(ep, [h["train_loss"] for h in history], "o-", label="train")
    a1.plot(ep, [h["val_loss"] for h in history], "s-", label="val")
    a1.set_xlabel("epoch"); a1.set_ylabel("loss"); a1.set_title("Loss"); a1.legend(); a1.grid(alpha=0.3)
    a2.plot(ep, [h["train_acc"] for h in history], "o-", label="train")
    a2.plot(ep, [h["val_acc"] for h in history], "s-", label="val")
    a2.set_xlabel("epoch"); a2.set_ylabel("accuracy"); a2.set_title("Accuracy"); a2.legend(); a2.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="KWS 学習ループ")
    parser.add_argument("--config", help="YAML config のパス")
    parser.add_argument("--device", help="cuda / cpu（明示指定）")
    parser.add_argument("--epochs", type=int, help="config の epochs を上書き")
    parser.add_argument("--run-name", dest="run_name", help="実験名（exp/<run_name>/ に保存）")
    parser.add_argument("--resume", action="store_true", help="exp/<run_name>/last.pt から再開")
    parser.add_argument("--no-wandb", action="store_true", help="wandb を使わない")
    args = parser.parse_args()

    cfg = load_config(
        args.config,
        {"device": args.device, "epochs": args.epochs, "run_name": args.run_name},
    )
    set_seed(cfg.seed)
    device = get_device(cfg.device)
    out_dir = Path("exp") / cfg.run_name
    # 既存の run ディレクトリは消さずタイムスタンプ付きで退避（resume 時はそのまま使う）
    if out_dir.exists() and not args.resume and any(out_dir.iterdir()):
        ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        backup = out_dir.with_name(f"{cfg.run_name}_backup_{ts}")
        out_dir.rename(backup)
        print(f"既存の {out_dir} を {backup} に退避した")
    out_dir.mkdir(parents=True, exist_ok=True)
    last_path, best_path = out_dir / "last.pt", out_dir / "best.pt"
    history_path = out_dir / "history.json"
    curve_path = out_dir / "loss_curve.png"
    log_path = out_dir / "train.log"

    def log(msg: str) -> None:
        """標準出力に出しつつ exp/<run>/train.log にも残す."""
        print(msg)
        with open(log_path, "a") as f:
            f.write(msg + "\n")

    if not args.resume:  # 新規開始ならログを初期化（resume 時は追記）
        log_path.write_text("")
    log(f"device: {device} / run: {cfg.run_name}")

    use_wandb = not args.no_wandb
    if use_wandb:
        try:
            import wandb

            wandb.init(project="kws-tutorial", name=cfg.run_name, config=asdict(cfg))
        except Exception as exc:  # noqa: BLE001
            log(f"[wandb] 無効化して続行（{exc}）")
            use_wandb = False

    loaders = get_dataloaders(cfg.data_root, cfg.batch_size, cfg.n_mels, cfg.num_workers)
    model = AudioCNN(n_classes=35, base=cfg.base).to(device)
    log(f"model params: {count_params(model):,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    scheduler = (
        torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)
        if cfg.scheduler == "cosine"
        else None
    )

    # config は学習前に書き出しておく（途中で落ちても何の設定で回したか残る）
    (out_dir / "config.json").write_text(
        json.dumps(asdict(cfg), indent=2, ensure_ascii=False)
    )

    history, best_acc, start_epoch = [], 0.0, 1
    if args.resume and last_path.exists():
        ckpt = load_checkpoint(last_path, model, optimizer, map_location=device)
        start_epoch = ckpt.get("epoch", 0) + 1
        best_acc = ckpt.get("best_acc", 0.0)
        if history_path.exists():
            history = json.loads(history_path.read_text())
        if scheduler is not None:  # scheduler の進行も再開位置に合わせる
            for _ in range(start_epoch - 1):
                scheduler.step()
        log(f"resume: epoch {start_epoch} から再開（best_acc {best_acc:.3f}）")

    for epoch in range(start_epoch, cfg.epochs + 1):
        tr_loss, tr_acc = run_epoch(
            model, loaders["train"], criterion, device, optimizer, desc=f"E{epoch} train",
        )
        va_loss, va_acc = run_epoch(
            model, loaders["val"], criterion, device, None, desc=f"E{epoch} val",
        )
        if scheduler is not None:
            scheduler.step()

        log(
            f"epoch {epoch:2d}: "
            f"train_loss {tr_loss:.3f} acc {tr_acc:.3f} | "
            f"val_loss {va_loss:.3f} acc {va_acc:.3f}"
        )
        row = {
            "epoch": epoch,
            "train_loss": tr_loss, "train_acc": tr_acc,
            "val_loss": va_loss, "val_acc": va_acc,
        }
        history.append(row)
        # history と loss 曲線は毎エポック更新（途中でも確認でき，中断時も残る）
        history_path.write_text(json.dumps(history, indent=2))
        save_loss_curve(history, curve_path)
        if use_wandb:
            wandb.log(row)

        # val が最良なら best を更新（先に更新して last にも正しい best_acc を残す）
        improved = va_acc > best_acc
        if improved:
            best_acc = va_acc
        # last.pt … 毎エポックの現在の重み（中断再開用）
        save_checkpoint(
            last_path, model, optimizer, epoch, best_acc, extra={"config": asdict(cfg)}
        )
        # best.pt … val accuracy が最良のエポックの重み
        if improved:
            save_checkpoint(
                best_path, model, optimizer, epoch, best_acc, extra={"config": asdict(cfg)}
            )

    log(f"best val acc: {best_acc:.3f}  →  {best_path}")
    log(f"artifacts: {out_dir}/ (last.pt, best.pt, history.json, loss_curve.png, train.log, config.json)")
    if use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
