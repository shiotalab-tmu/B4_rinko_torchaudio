import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import nbh
nbh.MODE = sys.argv[2] if len(sys.argv) > 2 else "full"
from nbh import md, code, sol, build

cells = [
md("""# 第2回 — モデルを作る（nn.Module）

第1回で作った log-mel ミニバッチ `(B, 1, n_mels, T)` を受け取り，35 クラスの logits を返す
小型 2D-CNN **AudioCNN** を作る．理論（CNN とは何か）は履修済み前提で，今日は **PyTorch でどう書くか**に集中．

**今日のゴール**: ランダム入力から 35 次元の logits が出る／**1 バッチを過学習**できる（配線が正しい証拠）．

対象ファイル: `src/kws/model.py`

> 参考: PyTorch 公式 "Build the Neural Network"
> https://docs.pytorch.org/tutorials/beginner/basics/buildmodel_tutorial.html"""),

code("""import os
from pathlib import Path
if Path.cwd().name == "notebooks":
    os.chdir("..")
import torch
from torch import nn
dev = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", dev)"""),

md("""## 1. nn.Module の書き方

PyTorch のモデルは `nn.Module` を継承して2つ書くだけ:
- `__init__`: 使う層を **属性として定義**（`self.conv = nn.Conv2d(...)` など）
- `forward`: 入力を層に通す **順伝播の流れ**

最小例で書き方を確認する（28次元 → 10次元の全結合）."""),

code("""class TinyNet(nn.Module):
    def __init__(self):
        super().__init__()                 # 必須: 親の初期化
        self.fc = nn.Linear(28, 10)        # 層を定義

    def forward(self, x):
        return self.fc(x)                  # 順伝播を書く

net = TinyNet()
print(net)
print("out shape:", net(torch.randn(4, 28)).shape)  # (4, 10)"""),

md("""## 2. ConvBlock を作る

畳み込みの1ブロックを `Conv2d → BatchNorm2d → ReLU → MaxPool2d` の順で組む．
`MaxPool2d(2)` で時間・周波数を半分に縮める．"""),

sol(
"""class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

    def forward(self, x):
        return self.block(x)

# 確認: (B,1,64,101) -> (B,16,32,50)
print(ConvBlock(1, 16)(torch.randn(2, 1, 64, 101)).shape)""",
"""class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        # TODO: Conv2d(3x3, padding=1, bias=False) -> BatchNorm2d -> ReLU -> MaxPool2d(2)
        self.block = nn.Sequential(
            ...
        )

    def forward(self, x):
        return self.block(x)

print(ConvBlock(1, 16)(torch.randn(2, 1, 64, 101)).shape)  # -> (2, 16, 32, 50)"""),

md("""## 3. AudioCNN を作る

`ConvBlock` を 4 段重ねて特徴抽出 → **Global Average Pooling（GAP）** で時間・周波数を1点に潰す →
`Linear` で 35 クラスへ．GAP を使うと入力の時間長 T に依存せず固定次元になる（可変長に強い）．

チャネル幅は `base` 起点で `base → base*2 → base*4 → base*4`."""),

sol(
"""class AudioCNN(nn.Module):
    def __init__(self, n_classes: int = 35, base: int = 32):
        super().__init__()
        self.features = nn.Sequential(
            ConvBlock(1, base),
            ConvBlock(base, base * 2),
            ConvBlock(base * 2, base * 4),
            ConvBlock(base * 4, base * 4),
        )
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(base * 4, n_classes)

    def forward(self, x):
        x = self.features(x)         # (B, C, h, w)
        x = self.gap(x).flatten(1)   # (B, C)
        return self.classifier(x)    # (B, n_classes)""",
"""class AudioCNN(nn.Module):
    def __init__(self, n_classes: int = 35, base: int = 32):
        super().__init__()
        # TODO: ConvBlock を 1->base, base->base*2, base*2->base*4, base*4->base*4 の4段
        self.features = nn.Sequential(
            ...
        )
        self.gap = nn.AdaptiveAvgPool2d(1)          # GAP: 空間を 1x1 に
        self.classifier = nn.Linear(base * 4, n_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.gap(x).flatten(1)   # (B, C)
        # TODO: 分類器に通して logits を返す
        return ..."""),

md("""## 4. 入出力 shape とパラメータ数を確認"""),

code("""model = AudioCNN(n_classes=35, base=32).to(dev)
dummy = torch.randn(8, 1, 64, 101, device=dev)
logits = model(dummy)
print("input :", tuple(dummy.shape))
print("logits:", tuple(logits.shape))   # (8, 35)

n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"trainable params: {n_params:,}")"""),

md("""## 5. 損失と optimizer（PyTorch での呼び出し方）

- 損失: 多クラス分類は `nn.CrossEntropyLoss`（内部で softmax + log + NLL．**logits をそのまま渡す**）
- 最適化: `torch.optim.Adam`．`model.parameters()` を渡し，`lr` を指定するだけ"""),

code("""criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
print(criterion)
print(optimizer)"""),

md("""## 6. sanity check: 1 バッチを過学習する → 今日のゴール

**同じ1ミニバッチ**を何十 step も学習して loss がほぼ 0 に落ちるか見る．落ちれば
「forward→loss→backward→step の配線が正しい」証拠．落ちなければラベルずれや勾配が流れない等のバグ．

第1回の `data.py` から1バッチ取ってくる（`data.py` を埋めていれば import できる）."""),

code("""from kws.data import get_dataloaders

loaders = get_dataloaders("data", batch_size=64, n_mels=64, num_workers=0)
feats, targets = next(iter(loaders["val"]))
feats, targets = feats.to(dev), targets.to(dev)
print("batch:", feats.shape, targets.shape)"""),

sol(
"""model = AudioCNN(n_classes=35, base=32).to(dev)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

model.train()
for step in range(60):
    optimizer.zero_grad()
    loss = criterion(model(feats), targets)
    loss.backward()
    optimizer.step()
    if step % 10 == 0 or step == 59:
        acc = (model(feats).argmax(1) == targets).float().mean().item()
        print(f"step {step:2d}: loss {loss.item():.4f}  acc {acc:.3f}")
print("\\nloss が 0 近くまで落ちれば配線OK！")""",
"""model = AudioCNN(n_classes=35, base=32).to(dev)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

model.train()
for step in range(60):
    # TODO: 学習の 4 ステップ（勾配リセット→loss計算→backward→step）
    optimizer.zero_grad()
    loss = criterion(..., ...)
    ...
    optimizer.step()
    if step % 10 == 0 or step == 59:
        acc = (model(feats).argmax(1) == targets).float().mean().item()
        print(f"step {step:2d}: loss {loss.item():.4f}  acc {acc:.3f}")"""),

md("""## 7. 整理 → `src/kws/model.py`

`ConvBlock` と `AudioCNN` を `src/kws/model.py` に整理する．解答は:

```bash
git checkout ans -- src/kws/model.py
```

次回（第3回）は，このモデルを **データ全体で学習するループ**を書く．"""),
]

build(cells, sys.argv[1])
