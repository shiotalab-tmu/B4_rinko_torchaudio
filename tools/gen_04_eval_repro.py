import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import nbh
nbh.MODE = sys.argv[2] if len(sys.argv) > 2 else "full"
from nbh import md, code, sol, build

cells = [
md("""# 第4回 — 評価と再現性

学習済みモデル（`exp/baseline/best.pt`）を **test セット**で評価する．
全体 accuracy だけでなく **混同行列**と **クラス別**の成績を見て，どこで間違えるかを考える．

**今日のゴール**: test accuracy と混同行列を出せる．

対象: `src/kws/evaluate.py`, `exp/baseline/`

> 前提: 第3回の本学習（`uv run python -m kws.train ...`）で `exp/baseline/best.pt` が作られていること．"""),

code("""import os
from pathlib import Path
if Path.cwd().name == "notebooks":
    os.chdir("..")
import json
import numpy as np
import torch
from kws.data import LABELS, get_dataloaders
from kws.model import AudioCNN
dev = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", dev, "| num classes:", len(LABELS))"""),

md("""## 1. checkpoint を読み込む

`best.pt` には `model`（重み）と `config`（学習設定）が入っている．config から `base` などを読んで
同じ形のモデルを作り，重みを流し込む．"""),

code("""ckpt_path = "exp/baseline/best.pt"
ckpt = torch.load(ckpt_path, map_location=dev, weights_only=False)
cfg = ckpt.get("config", {})
base, n_mels = cfg.get("base", 32), cfg.get("n_mels", 64)
print("best epoch:", ckpt.get("epoch"), "| val best_acc:", round(ckpt.get("best_acc", 0), 4))

model = AudioCNN(n_classes=len(LABELS), base=base).to(dev)
model.load_state_dict(ckpt["model"])
model.eval()
print("loaded:", ckpt_path)"""),

md("""## 2. test セットを推論

評価では勾配は不要なので `@torch.no_grad()`．各バッチで logits の `argmax` が予測クラス．
正解ラベルと予測ラベルを全部集める．"""),

sol(
"""@torch.no_grad()
def predict(model, loader, device):
    ys, ps = [], []
    for feats, targets in loader:
        feats = feats.to(device)
        logits = model(feats)
        ps.append(logits.argmax(1).cpu())
        ys.append(targets)
    return torch.cat(ys).numpy(), torch.cat(ps).numpy()

loaders = get_dataloaders("data", batch_size=256, n_mels=n_mels, num_workers=0)
y_true, y_pred = predict(model, loaders["test"], dev)
print("predicted:", y_pred.shape[0], "samples")""",
"""@torch.no_grad()
def predict(model, loader, device):
    ys, ps = [], []
    for feats, targets in loader:
        feats = feats.to(device)
        logits = model(feats)
        # TODO: 予測クラス（logits の argmax）と正解を集める
        ps.append(...)
        ys.append(targets)
    return torch.cat(ys).numpy(), torch.cat(ps).numpy()

loaders = get_dataloaders("data", batch_size=256, n_mels=n_mels, num_workers=0)
y_true, y_pred = predict(model, loaders["test"], dev)
print("predicted:", y_pred.shape[0], "samples")"""),

md("""## 3. accuracy とクラス別レポート

全体 accuracy と，`scikit-learn` の `classification_report`（precision / recall / F1 をクラス別に）."""),

code("""from sklearn.metrics import classification_report

acc = float((y_true == y_pred).mean())
print(f"test accuracy: {acc:.4f}\\n")
print(classification_report(y_true, y_pred, target_names=LABELS, digits=3))"""),

md("""## 4. 混同行列（confusion matrix）

行=正解，列=予測．対角が濃ければ良い．対角から外れた濃いマスが「混同しているクラス対」．"""),

code("""from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

cm = confusion_matrix(y_true, y_pred)
fig, ax = plt.subplots(figsize=(10, 9))
im = ax.imshow(cm, cmap="Blues")
ax.set_xticks(range(len(LABELS))); ax.set_xticklabels(LABELS, rotation=90, fontsize=6)
ax.set_yticks(range(len(LABELS))); ax.set_yticklabels(LABELS, fontsize=6)
ax.set_xlabel("predicted"); ax.set_ylabel("true")
ax.set_title(f"Confusion Matrix (test acc = {acc:.3f})")
fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04); fig.tight_layout(); plt.show()"""),

md("""## 5. どのクラス対が混同しやすい？

混同行列の非対角成分が大きい順に並べると，似た音（例: go/no, three/tree など）が見える．"""),

code("""cm0 = cm.copy()
np.fill_diagonal(cm0, 0)
pairs = [(LABELS[i], LABELS[j], cm0[i, j])
         for i in range(len(LABELS)) for j in range(len(LABELS)) if cm0[i, j] > 0]
pairs.sort(key=lambda t: t[2], reverse=True)
print("混同が多い (true -> pred) 上位:")
for t, p, n in pairs[:10]:
    print(f"  {t:>8} -> {p:<8} : {n}")"""),

md("""## 6. 再現性（reproducibility）

実験は **同じ条件なら同じ結果**になるようにしておく．`kws.utils.set_seed` は Python / NumPy /
PyTorch の乱数を固定し，cudnn を deterministic にする．config と `uv.lock` を残せば環境ごと再現できる．"""),

code("""from kws.utils import set_seed
set_seed(42)
a = torch.randn(3)
set_seed(42)
b = torch.randn(3)
print("seed 固定で同じ乱数:", torch.allclose(a, b))"""),

md("""## 7. run ディレクトリ規約と整理 → `src/kws/evaluate.py`

1つの実験の成果物は `exp/<run_name>/` にまとまる:

```
exp/baseline/
├── config.json          # 学習設定
├── history.json         # epoch ごとの loss/acc
├── train.log            # 学習の出力ログ（毎エポックの値）
├── loss_curve.png       # loss/acc 曲線（学習中に毎エポック更新）
├── last.pt / best.pt    # checkpoint（現在の重み / val最良）
├── confusion_matrix.png # 混同行列の図
└── test_metrics.json    # test accuracy + クラス別 P/R/F1（全部保存）
```

評価ロジックは `src/kws/evaluate.py` にまとまっている:

```bash
uv run python -m kws.evaluate --ckpt exp/baseline/best.pt --device cuda
```

解答: `git checkout ans -- src/kws/evaluate.py`．次回（第5回）は **自分の工夫で精度を上げる**ワークショップ．"""),
]

build(cells, sys.argv[1])
