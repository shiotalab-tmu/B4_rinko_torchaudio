"""test セットでの評価：accuracy / confusion matrix / per-class（classification report）.

使い方:
    uv run python -m kws.evaluate --ckpt exp/baseline/best.pt --device cuda
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")  # 画面の無い環境でも図を保存できるように
import matplotlib.pyplot as plt  # noqa: E402
from sklearn.metrics import classification_report, confusion_matrix  # noqa: E402

from kws.data import LABELS, get_dataloaders  # noqa: E402
from kws.model import AudioCNN  # noqa: E402
from kws.utils import get_device  # noqa: E402


@torch.no_grad()
def predict(model: torch.nn.Module, loader, device: torch.device):
    """loader 全体について (正解ラベル, 予測ラベル) の numpy 配列を返す."""
    model.eval()
    ys, ps = [], []
    for feats, targets in loader:
        feats = feats.to(device)
        logits = model(feats)
        ps.append(logits.argmax(1).cpu())
        ys.append(targets.cpu())
    return torch.cat(ys).numpy(), torch.cat(ps).numpy()


def plot_confusion(cm: np.ndarray, acc: float, path: Path) -> None:
    """confusion matrix を画像に保存する."""
    fig, ax = plt.subplots(figsize=(11, 10))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(LABELS)))
    ax.set_xticklabels(LABELS, rotation=90, fontsize=6)
    ax.set_yticks(range(len(LABELS)))
    ax.set_yticklabels(LABELS, fontsize=6)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(f"Confusion Matrix (test acc = {acc:.3f})")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="KWS 評価")
    parser.add_argument("--ckpt", required=True, help="checkpoint (best.pt) のパス")
    parser.add_argument("--device", help="cuda / cpu（明示指定）")
    parser.add_argument("--data-root", dest="data_root", default="data")
    parser.add_argument("--batch-size", dest="batch_size", type=int, default=256)
    args = parser.parse_args()

    device = get_device(args.device)
    # 自前で保存した checkpoint（config dict を含む）なので weights_only=False を明示．
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    cfg = ckpt.get("config", {})
    base = cfg.get("base", 32)
    n_mels = cfg.get("n_mels", 64)

    model = AudioCNN(n_classes=len(LABELS), base=base).to(device)
    model.load_state_dict(ckpt["model"])

    loaders = get_dataloaders(
        args.data_root, args.batch_size, n_mels=n_mels, num_workers=4
    )
    y_true, y_pred = predict(model, loaders["test"], device)

    acc = float((y_true == y_pred).mean())
    print(f"test accuracy: {acc:.4f}\n")
    print(classification_report(y_true, y_pred, target_names=LABELS, digits=3))

    # 評価情報は全部残す：accuracy / クラス別 P-R-F1 / macro・weighted avg（dict）
    report = classification_report(
        y_true, y_pred, target_names=LABELS, digits=4, output_dict=True
    )
    cm = confusion_matrix(y_true, y_pred)
    out_dir = Path(args.ckpt).parent
    np.save(out_dir / "confusion_matrix.npy", cm)
    (out_dir / "test_metrics.json").write_text(
        json.dumps(
            {
                "accuracy": acc,
                "num_samples": int(y_true.shape[0]),
                "ckpt": str(args.ckpt),
                "classification_report": report,
            },
            indent=2,
        )
    )
    plot_confusion(cm, acc, out_dir / "confusion_matrix.png")
    print(f"\nsaved: {out_dir / 'confusion_matrix.png'}")
    print(f"saved: {out_dir / 'test_metrics.json'}（全クラスの P/R/F1 を含む）")


if __name__ == "__main__":
    main()
