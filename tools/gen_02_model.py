import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import nbh
nbh.MODE = sys.argv[2] if len(sys.argv) > 2 else "full"
from nbh import md, code, sol, build

cells = [

# ── タイトル ────────────────────────────────────────────────────────────────
md("""# 第2回 — モデルを作る（nn.Module）

第1回で作った log-mel ミニバッチ `(B, 1, n_mels, T')` を受け取り，35 クラスの logits を返す
小型 2D-CNN **AudioCNN** を作る．

**今日のゴール**: ランダム入力から `(B, 35)` の logits が出る，かつ **1 バッチを過学習できる**（配線が正しい証拠）．

対象ファイル: `src/kws/model.py`

> 参考: PyTorch 公式 "Build the Neural Network"
> https://docs.pytorch.org/tutorials/beginner/basics/buildmodel_tutorial.html"""),

# ── セットアップ ────────────────────────────────────────────────────────────
code("""import os, sys
from pathlib import Path
if Path.cwd().name == "notebooks":
    os.chdir("..")
sys.path.insert(0, "src")

import torch
from torch import nn

dev = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", dev)"""),

# ── ハンズオン①: nn.Module 基礎 ────────────────────────────────────────────
md("""## ハンズオン①: nn.Module の書き方（10分）

PyTorch のモデルは `nn.Module` を継承して2つのメソッドを書くだけ:

- `__init__`: 使う層を **属性として定義**（`self.conv = nn.Conv2d(...)` など）
- `forward`: 入力を層に通す **順伝播の流れ**

ここで定義した層のパラメータが `model.parameters()` で取れる．
`model(x)` と呼ぶと内部で `forward(x)` が走る．"""),

code("""class TinyNet(nn.Module):
    def __init__(self):
        super().__init__()            # 必須: 親の初期化
        self.fc = nn.Linear(28, 10)   # 層を定義

    def forward(self, x):
        return self.fc(x)             # 順伝播を書く

net = TinyNet()
print(net)
print("out:", net(torch.randn(4, 28)).shape)  # (4, 10)"""),

md("""### model.to(device)

モデルも Tensor と同じく `.to(device)` で GPU に移せる．
**モデルと入力は同じ device に揃える**（揃えないとエラー）．"""),

code("""net = TinyNet().to(dev)
x = torch.randn(4, 28, device=dev)
print("out:", net(x).shape)"""),

# ── ハンズオン②: ConvBlock ──────────────────────────────────────────────────
md("""## ハンズオン②: ConvBlock を作る（15分）

畳み込みの1ブロック: `Conv2d(3×3)` → `BatchNorm2d` → `ReLU` → `MaxPool2d(2)`

各層の役割:
- `Conv2d(kernel_size=3, padding=1, bias=False)`: 3×3 畳み込みで空間サイズを保つ（`bias=False` は BN と組み合わせる定番）
- `BatchNorm2d`: チャネルごとに正規化して学習を安定させる
- `ReLU`: 活性化関数（`max(0, x)`）
- `MaxPool2d(2)`: 2×2 窓で最大値を取り，**空間サイズを半分**に縮める"""),

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

# 確認: (B,1,64,101) → (B,16,32,50)（MaxPool で H,W 半減）
print(ConvBlock(1, 16)(torch.randn(2, 1, 64, 101)).shape)""",

"""class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        # TODO: Conv2d(3x3, padding=1, bias=False) → BatchNorm2d → ReLU → MaxPool2d(2)
        self.block = nn.Sequential(
            ...
        )

    def forward(self, x):
        return self.block(x)

# 確認: (B,1,64,101) → (B,16,32,50)
print(ConvBlock(1, 16)(torch.randn(2, 1, 64, 101)).shape)"""),

# ── ハンズオン②: AudioCNN ──────────────────────────────────────────────────
md("""## ハンズオン②: AudioCNN を作る（15分）

`ConvBlock` を**3段**重ねて特徴抽出 → **Global Average Pooling (GAP)** → `Linear` で 35 クラスへ．

GAP (`AdaptiveAvgPool2d(1)`) の役割: 空間方向を1点に潰して固定長ベクトルにする．
入力の時間長 T' が変わっても GAP 後は必ず `(B, C)` の固定形状になる．

チャネル幅は `base` 起点: `1 → base → base*2 → base*4`"""),

sol(
"""class AudioCNN(nn.Module):
    def __init__(self, n_classes: int = 35, base: int = 32):
        super().__init__()
        # ConvBlock 3段: 1→base, base→base*2, base*2→base*4
        self.features = nn.Sequential(
            ConvBlock(1, base),
            ConvBlock(base, base * 2),
            ConvBlock(base * 2, base * 4),
        )
        self.gap = nn.AdaptiveAvgPool2d(1)          # GAP: 空間を 1x1 に
        self.classifier = nn.Linear(base * 4, n_classes)

    def forward(self, x):
        x = self.features(x)            # (B, C, h, w)
        x = self.gap(x).flatten(1)      # (B, C)
        return self.classifier(x)       # (B, n_classes)

# 確認: (B,1,64,101) → (B,35)
print(AudioCNN()(torch.randn(8, 1, 64, 101)).shape)""",

"""class AudioCNN(nn.Module):
    def __init__(self, n_classes: int = 35, base: int = 32):
        super().__init__()
        # TODO: ConvBlock を 1→base, base→base*2, base*2→base*4 の3段
        self.features = nn.Sequential(
            ...
        )
        self.gap = nn.AdaptiveAvgPool2d(1)          # GAP: 空間を 1x1 に
        self.classifier = nn.Linear(base * 4, n_classes)

    def forward(self, x):
        x = self.features(x)            # (B, C, h, w)
        x = self.gap(x).flatten(1)      # (B, C)
        # TODO: 分類器に通して logits を返す
        return ...

# 確認: (B,1,64,101) → (B,35)
print(AudioCNN()(torch.randn(8, 1, 64, 101)).shape)"""),

# ── ハンズオン③: shape 追跡 ────────────────────────────────────────────────
md("""## ハンズオン③: shape を1段ずつ追跡する（動く例）

各 `ConvBlock` で `MaxPool2d(2)` が空間を半分にすることを数値で確認する．"""),

code("""x = torch.randn(2, 1, 64, 101, device=dev)
model = AudioCNN(n_classes=35, base=32).to(dev)

print("入力              :", tuple(x.shape))
x1 = model.features[0](x)
print("ConvBlock(1,32)   :", tuple(x1.shape))   # (B, 32, 32, 50)
x2 = model.features[1](x1)
print("ConvBlock(32,64)  :", tuple(x2.shape))   # (B, 64, 16, 25)
x3 = model.features[2](x2)
print("ConvBlock(64,128) :", tuple(x3.shape))   # (B, 128, 8, 12)
x4 = model.gap(x3).flatten(1)
print("GAP + flatten     :", tuple(x4.shape))   # (B, 128)
out = model.classifier(x4)
print("Linear(128, 35)   :", tuple(out.shape))  # (B, 35)"""),

md("""### パラメータ数を確認する"""),

code("""n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"trainable params: {n_params:,}")"""),

md("""### base を変えるとパラメータ数はどう変わるか

Conv の重みは `in_ch × out_ch × k × k` なので，base を2倍にするとパラメータ数は約4倍になる．"""),

code("""for b in [16, 32, 64]:
    m = AudioCNN(base=b)
    n = sum(p.numel() for p in m.parameters() if p.requires_grad)
    print(f"base={b:2d}: {n:>8,} params")"""),

# ── ハンズオン③: CrossEntropyLoss 説明 ────────────────────────────────────
md("""## ハンズオン③: 損失関数 CrossEntropyLoss（動く例）

多クラス分類には `nn.CrossEntropyLoss` を使う．
内部で **log_softmax + NLL** を計算するので，**forward で softmax をかけてはいけない**．
softmax を二重にかけると勾配が潰れ，学習がうまく進まなくなる．

ランダム重みのモデルは各クラスをほぼ等確率で予測する → 初期 loss ≈ `-log(1/35) = ln(35) ≈ 3.56`"""),

code("""import math
criterion = nn.CrossEntropyLoss()

dummy_logits = torch.randn(8, 35)   # model の出力（raw logits）
dummy_targets = torch.randint(0, 35, (8,))
loss = criterion(dummy_logits, dummy_targets)
print(f"初期 loss: {loss.item():.3f}  (ln(35) = {math.log(35):.3f})")"""),

# ── ハンズオン③: 1バッチ過学習 sanity check ───────────────────────────────
md("""## ハンズオン③: 1バッチ過学習 sanity check（動く例）

**同じ1ミニバッチ**を何百 step も学習して loss が ~0 に落ちるか見る．
落ちれば「forward→loss→backward→step の配線が正しい」証拠（Karpathy sanity check）．

学習ループは**動く例として提供**する（受講者がループを自分で書くのは第3回）．
ここではランダムテンソルを使う（data に依存しない）．"""),

code("""from kws.utils import set_seed

set_seed(42)

model_sanity = AudioCNN(n_classes=35, base=32).to(dev)
optimizer = torch.optim.Adam(model_sanity.parameters(), lr=1e-3)

dummy_feats = torch.randn(8, 1, 64, 101, device=dev)
dummy_labels = torch.randint(0, 35, (8,), device=dev)

model_sanity.train()
for step in range(200):
    optimizer.zero_grad()
    logits = model_sanity(dummy_feats)
    loss = criterion(logits, dummy_labels)
    loss.backward()
    optimizer.step()
    if step % 50 == 0 or step == 199:
        print(f"step {step:3d}: loss {loss.item():.4f}")

print("\\nloss が 0 近くまで落ちれば配線 OK!")"""),

# ── check_model ────────────────────────────────────────────────────────────
md("""## check_model（PASS で完成）

出力 shape・初期 loss・勾配の存在・1バッチ過学習の4点を assert で確認する．
全て通れば model 完成．"""),

code("""\"\"\"model の確認: 出力 shape・初期 loss・grad 存在・1バッチ過学習を assert する.\"\"\"
import math, torch
from torch import nn
from kws.utils import set_seed

set_seed(42)

N_MELS, T_PRIME, N_CLASSES, B = 64, 101, 35, 8

# --- 1. 出力 shape ---
model_check = AudioCNN(n_classes=N_CLASSES, base=32)
dummy = torch.randn(B, 1, N_MELS, T_PRIME)
logits = model_check(dummy)
assert logits.shape == (B, N_CLASSES), f"出力 shape が (B,35) でない: {logits.shape}"

# --- 2. 初期 loss ≈ ln(35) ≈ 3.56 ---
criterion_check = nn.CrossEntropyLoss()
targets = torch.randint(0, N_CLASSES, (B,))
init_loss = criterion_check(logits, targets).item()
expected = math.log(N_CLASSES)
assert abs(init_loss - expected) < 0.5, f"初期 loss {init_loss:.3f} が ln(35)≈{expected:.2f} から離れすぎ"

# --- 3. backward 後に grad が存在 ---
loss_tmp = criterion_check(model_check(dummy), targets)
loss_tmp.backward()
for name, p in model_check.named_parameters():
    assert p.grad is not None, f"{name} の grad が None（勾配が流れていない）"

# --- 4. 1バッチ過学習（200 step で loss → ~0） ---
set_seed(42)
model2 = AudioCNN(n_classes=N_CLASSES, base=32)
opt2 = torch.optim.Adam(model2.parameters(), lr=1e-3)
feats2 = torch.randn(B, 1, N_MELS, T_PRIME)
labels2 = torch.randint(0, N_CLASSES, (B,))
model2.train()
for _ in range(200):
    opt2.zero_grad()
    out2 = model2(feats2)
    l2 = criterion_check(out2, labels2)
    l2.backward()
    opt2.step()
final_loss = l2.item()
assert final_loss < 0.05, f"200 step 後の loss {final_loss:.4f} が下がりきらない（forward にバグ？）"

print(f"check_model PASS: logits {tuple(logits.shape)}, init_loss {init_loss:.2f}, overfit_loss {final_loss:.4f}")"""),

# ── まとめ ────────────────────────────────────────────────────────────────
md("""## まとめ

今日のポイント:
- `nn.Module` は `__init__` で層を定義，`forward` で繋ぐだけ
- `ConvBlock`（Conv→BN→ReLU→MaxPool）を積み上げて `AudioCNN` を構成
- GAP（`AdaptiveAvgPool2d(1)`）で空間を1点に潰すと，T' によらず固定長になる
- 1バッチ過学習（loss → ~0）は「配線が正しい」証拠 = Karpathy sanity check

### 宿題①: src/kws/model.py に移植する

notebook で書いた `ConvBlock`・`AudioCNN` を `src/kws/model.py` の TODO に埋める．

`sys.path.insert(0, "src")` のあと `from kws.model import AudioCNN` が通ることを確認．

### 宿題②: 第3回（学習ループ）の予習を分担で1スライド

第3回では今日の `zero_grad → forward → loss → backward → step` の5行を
**受講者自身が書く**（今日の sanity check で見た流れがそのままメインテーマになる）．"""),

]

build(cells, sys.argv[1])
