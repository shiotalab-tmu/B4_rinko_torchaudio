import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import nbh
nbh.MODE = sys.argv[2] if len(sys.argv) > 2 else "full"
from nbh import md, code, sol, build

cells = [

# ── タイトル ──────────────────────────────────────────────────────────────────
md("""# 第3回 — 学習ループ

データ（第1回）とモデル（第2回）が揃ったので，**学習ループ**を書く．ここがチュートリアルの山場．

**今日のゴール**: `check_train` が PASS する（train loss が下がる・val acc がチャンスレート超え）．

対象: `src/kws/train.py`

関連リンク:
- PyTorch "Optimizing Model Parameters": https://docs.pytorch.org/tutorials/beginner/basics/optimization_tutorial.html
- `nn.CrossEntropyLoss`: https://docs.pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html
- `torch.optim.Adam`: https://docs.pytorch.org/docs/stable/generated/torch.optim.Adam.html"""),

# ── セットアップ ───────────────────────────────────────────────────────────────
code("""import os
from pathlib import Path
if Path.cwd().name == "notebooks":
    os.chdir("..")
import torch
from torch import nn
print("device:", "cuda" if torch.cuda.is_available() else "cpu")"""),

# ── ハンズオン① ────────────────────────────────────────────────────────────────
md("""## ハンズオン① — CrossEntropyLoss を理解する（10分）

### CrossEntropyLoss に渡すもの

`CrossEntropyLoss` は **生の logits `(B, C)`** と **正解ラベル `(B,)`（整数 index・`long`）** を受け取る．
第1回で作った `label_to_index` の整数がそのままここに来る．"""),

code("""criterion = nn.CrossEntropyLoss()
logits = torch.randn(4, 35)            # 4サンプル，35クラス
labels = torch.tensor([0, 5, 12, 34])  # 正解の index
loss = criterion(logits, labels)
print(f"loss: {loss.item():.4f}")"""),

md("""### softmax をかけてはいけない

`CrossEntropyLoss` は内部で `log_softmax` を行う．forward の出力に `softmax` をかけてから渡すと二重になり，
勾配が潰れて学習がうまく進まなくなる．完全にエラーにはならないが性能が大幅に劣化する．

### 初期 loss ≈ ln(35) ≈ 3.56

ランダム重みなら各クラスをほぼ均等に予測する → 初期 loss は `-log(1/35)` 付近になる．
第2回の sanity check で確認済みの数値．

### 手を動かす確認

正解クラスの logit を大きくすると loss はどう変わるか，予想してから実行してみよう．"""),

code("""# 正解クラス（index=0）の logit だけ大きくした場合
logits_good = logits.clone()
logits_good[0, 0] = 10.0  # サンプル0の正解クラスを強制的に大きく
loss_good = criterion(logits_good, labels)
print(f"loss (ランダム):    {loss.item():.4f}")
print(f"loss (正解強調後): {loss_good.item():.4f}")
# → 正解クラスの logit を大きくすると loss が下がる"""),

md("""### model.eval() と torch.no_grad() の使い分け

- `model.train()` / `model.eval()` — **モデル内部の挙動**を切り替える（BatchNorm は train 時にバッチ統計を更新し，eval 時に蓄積した running_mean / running_var を使う）．
- `torch.no_grad()` — **autograd**（勾配計算）を off にする．評価時は勾配が不要なのでメモリを節約できる．

eval 時は**両方**設定する必要がある（別の仕組みなので片方だけでは不十分）．"""),

# ── ハンズオン② ────────────────────────────────────────────────────────────────
md("""## ハンズオン② — 学習ループを書く（25分）

ここが今日の山場．`run_epoch` 関数の**学習の3ステップ**を自分で書く．

### 全体構造

- `optimizer` を渡すと学習モード，渡さないと評価モード．同じ関数で train / val を兼ねる．
- `model.train(is_train)` で BatchNorm・Dropout の挙動を切り替える．
- `torch.set_grad_enabled(is_train)` で勾配計算の on/off を切り替える．

### 書く部分

```python
if is_train:
    # 1. 勾配リセット
    # 2. 誤差逆伝播
    # 3. パラメータ更新
```

#### 各ステップの意味

- `zero_grad()` — 前の step の勾配をリセットする．PyTorch は勾配を**足し込む**ので，リセットしないと前の step の勾配が残ってしまう．
- `loss.backward()` — 計算グラフを辿って各パラメータの勾配を計算する（誤差逆伝播法がこの1行）．
- `optimizer.step()` — 計算された勾配を使ってパラメータを更新する（Adam の更新式がこの1行）．

### よくあるエラー早見表

| 症状 | よくある原因 |
|---|---|
| loss が下がらない（3.56 付近で横ばい） | `zero_grad` / `backward` / `step` のどれかが抜けている |
| loss が `nan` になる | forward で softmax をかけた logits を `CrossEntropyLoss` に渡している（二重 softmax） |
| `RuntimeError: element 0 ... does not require grad` | `torch.no_grad()` 内で backward を呼んでいる |
| GPU メモリ不足 | `batch_size` を小さくする（128 や 64 で試す） |
| val なのに loss が下がり続ける | val 呼び出しで `optimizer=None` を渡し忘れている |"""),

sol(
# full（解答版）
"""def run_epoch(model, loader, criterion, device, optimizer=None, desc=""):
    is_train = optimizer is not None
    model.train(is_train)
    total, correct, loss_sum = 0, 0, 0.0

    for feats, targets in loader:
        feats, targets = feats.to(device), targets.to(device)
        with torch.set_grad_enabled(is_train):
            logits = model(feats)              # forward
            loss = criterion(logits, targets)  # loss
        if is_train:
            optimizer.zero_grad()  # 勾配をリセット
            loss.backward()        # 勾配を計算
            optimizer.step()       # パラメータ更新
        loss_sum += loss.item() * targets.size(0)
        correct += (logits.argmax(1) == targets).sum().item()
        total += targets.size(0)

    return loss_sum / total, correct / total""",
# blank（穴埋め版）
"""def run_epoch(model, loader, criterion, device, optimizer=None, desc=""):
    is_train = optimizer is not None
    model.train(is_train)
    total, correct, loss_sum = 0, 0, 0.0

    for feats, targets in loader:
        feats, targets = feats.to(device), targets.to(device)
        with torch.set_grad_enabled(is_train):
            logits = model(feats)              # forward
            loss = criterion(logits, targets)  # loss
        if is_train:
            # TODO: 学習の3ステップを書く
            # 1. 勾配リセット
            # 2. 誤差逆伝播
            # 3. パラメータ更新
            ...
        loss_sum += loss.item() * targets.size(0)
        correct += (logits.argmax(1) == targets).sum().item()
        total += targets.size(0)

    return loss_sum / total, correct / total"""),

md("""### 書けたら動作確認

1 epoch だけ回して loss と accuracy を確認する．
train loss が初期（≈3.56）から下がっていれば学習が回っている証拠．"""),

code("""import sys
sys.path.insert(0, "src")
from kws.data import get_dataloaders
from kws.model import AudioCNN
from kws.utils import set_seed

set_seed(42)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
loaders = get_dataloaders('data', batch_size=256, n_mels=64, num_workers=0)
model = AudioCNN(n_classes=35, base=32).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# 1 epoch だけ回して loss と accuracy を確認
tr_loss, tr_acc = run_epoch(model, loaders['train'], criterion, device, optimizer)
print(f"train: loss {tr_loss:.3f}, acc {tr_acc:.3f}")
# 1 epoch で 2.5〜3.0 程度まで下がれば順調"""),

# ── ハンズオン③ ────────────────────────────────────────────────────────────────
md("""## ハンズオン③ — 学習を回して確認する（25分）

②で書いた `run_epoch` を使って 5 epoch 学習を回し，print 出力で確認する．

②の動作確認で 1 epoch 分学習が進んだ model をそのまま使い続ける．"""),

code("""for epoch in range(1, 6):  # 5 epoch（当日は時間の都合で少なめ）
    tr_loss, tr_acc = run_epoch(model, loaders['train'], criterion, device, optimizer)
    va_loss, va_acc = run_epoch(model, loaders['val'], criterion, device, None)
    print(f"epoch {epoch}: train loss {tr_loss:.3f} acc {tr_acc:.3f} | val loss {va_loss:.3f} acc {va_acc:.3f}")"""),

md("""### test accuracy

学習が終わったら test で 1 回だけ評価して accuracy を出す（confusion matrix 等は第4回）．"""),

code("""te_loss, te_acc = run_epoch(model, loaders['test'], criterion, device, None)
print(f"test accuracy: {te_acc:.3f}")"""),

md("""### ここで気づく問題: ログが残らない

今のループは print しかしていないので，後から振り返れない:

- print で見れるのは今この瞬間だけ．ノートブックを閉じたら消える．
- 学習率や epoch 数を変えて比較したくても，前回の結果が消えている．
- 長時間の学習を走らせて後から結果を確認したい場合に困る．

history を JSON に書き出す？ loss 曲線を matplotlib で描く？ → 第4回でやる．"""),

# ── check_train ────────────────────────────────────────────────────────────────
md("""## check_train

学習ループが正しく動いているか確認する．**3 epoch 回して loss が下がる・val acc がチャンスレート超え**を assert する．

PASS の例:
```
check_train PASS: train_loss 3.401→2.156, val_acc 0.312, test_acc 0.298
```"""),

code("""# check_train — 学習ループが正しく動いているかを確認する
import sys, torch
from torch import nn

sys.path.insert(0, "src")
from kws.data import get_dataloaders
from kws.model import AudioCNN
from kws.utils import set_seed

set_seed(42)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
loaders = get_dataloaders('data', batch_size=256, n_mels=64, num_workers=0)
model = AudioCNN(n_classes=35, base=32).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# --- 1. 3 epoch 回して loss が下がるか ---
losses = []
for epoch in range(1, 4):
    tr_loss, tr_acc = run_epoch(model, loaders['train'], criterion, device, optimizer)
    losses.append(tr_loss)
assert losses[-1] < losses[0], f"train loss が下がっていない: {losses}"

# --- 2. val accuracy がチャンスレート(1/35≈2.9%)を超えるか ---
va_loss, va_acc = run_epoch(model, loaders['val'], criterion, device, None)
assert va_acc > 0.10, f"val accuracy {va_acc:.3f} が低すぎる"

# --- 3. test accuracy が出るか ---
te_loss, te_acc = run_epoch(model, loaders['test'], criterion, device, None)
assert te_acc > 0.10, f"test accuracy {te_acc:.3f} が低すぎる"

print(f"check_train PASS: train_loss {losses[0]:.3f}→{losses[-1]:.3f}, val_acc {va_acc:.3f}, test_acc {te_acc:.3f}")"""),

# ── まとめ ────────────────────────────────────────────────────────────────────
md("""## まとめ

### 今日やったこと

- `CrossEntropyLoss` が生の logits を受け取り，内部で log_softmax + NLL を行うことを確認した．
- `run_epoch` の学習3ステップ（`zero_grad` → `backward` → `step`）を自分で書いた．
- train / val を分けて 5 epoch 学習を回し，print で loss の推移を確認した．
- `check_train` が PASS した（学習ループが正しく動いている）．

### 宿題

**① 実装を `src/kws/train.py` に移植して整理**

当日 notebook で書いた `run_epoch` を `src/kws/train.py` の TODO に埋め，`check_train` を PASS させる．
移植後は CLI で本格学習を回す:

```bash
uv run python -m kws.train --config configs/baseline.yaml --device cuda
```

`exp/baseline/` に `last.pt`・`best.pt` が出力される．この `best.pt` が第4回の入力になる．
`train.py` は argparse + YAML config の CLI になっている．

**② 第4回（評価・チューニング）の予習を分担で1スライド**

各自「分かったこと1点 ＋ 疑問1点」を持ち寄る．

### 第4回への引き継ぎ

第4回では学習の記録・可視化と，テスト評価・チューニングを扱う:

- 今日の「ログが残らない」問題を解決する（学習経過の記録と loss 曲線の可視化）．
- 学習曲線を読んで overfit / underfit を判断する．
- confusion matrix で「どのクラスを間違えるか」を分析する．"""),

]

build(cells, sys.argv[1])
