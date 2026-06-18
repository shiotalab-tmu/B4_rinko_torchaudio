# 第2回 — model（`session2_model.md`）

> 各回の地図は `docs/sessions/index.md`．本ファイルは第2回の超詳細（読めば誰でも同じノートブックを起こせるレベルを目指す）．
> 対象モジュール：`src/kws/model.py`．完成の確認＝`check_model` が PASS．

## この回の作り方の方針（重要）

- session1 と同じく**当日は notebook の中だけで完結**させる．`src/kws/model.py` への移植は**宿題**．
- **答えになる実装**（`ConvBlock`・`AudioCNN` の構造）は**抽象例**（骨組み＋流れコメント＋`...`）で配り，受講者が中身を書く．
- **答えにならない部分**（`nn.Module` の書き方・device の載せ方・shape の追跡・1バッチ過学習の体験）は**動く例**を見せて受講者も打って確認．
- 1バッチ過学習で使う**学習ループ（`zero_grad`→`forward`→`loss`→`backward`→`step`）は第3回の穴埋め対象**なので，ここでは**動く例として提供**する（受講者にループ自体を書かせるのは第3回）．目的はあくまで「forward の実装にバグがないか・勾配がちゃんと流れるかの sanity check」．

## ねらい

- `nn.Module` の書き方（`__init__` で層を定義・`forward` で繋ぐ）を理解し，**自分で書けるようになる**．
- `ConvBlock`（Conv2d → BN → ReLU → MaxPool）と `AudioCNN`（ConvBlock 3段 ＋ GAP ＋ Linear）を自分の手で組み立てる．
- shape の変化を**各層ごとに追跡**し，入力 `(B,1,n_mels,T')` から出力 `(B,35)` logits までの流れを把握する．
- 1バッチ過学習（loss → ~0）で**実装の正しさ**を自分で確認できるようになる（Karpathy sanity check）．
- 達成目標＝**`check_model` が PASS**（出力 shape・初期 loss・過学習・grad 存在の assert が通る）．

## 関連リンク

- PyTorch "Build the Neural Network"（`nn.Module` の書き方・公式）：https://docs.pytorch.org/tutorials/beginner/basics/buildmodel_tutorial.html
  - 日本語訳：https://yutaroogawa.github.io/pytorch_tutorials_jp/（同じ章）
- `nn.Conv2d` のドキュメント：https://docs.pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
- `nn.BatchNorm2d`：https://docs.pytorch.org/docs/stable/generated/torch.nn.BatchNorm2d.html
- `nn.AdaptiveAvgPool2d`（GAP）：https://docs.pytorch.org/docs/stable/generated/torch.nn.AdaptiveAvgPool2d.html
- 日本語の補助記事（公式以外）：
  - CNN 構築：[PyTorchでCNNを徹底解説](https://qiita.com/mathlive/items/8e1f9a8467fff8dfd03c)（Qiita・nn.Module 継承・Conv2d・forward 定義を逐行コメント付きで説明）
  - CNN 画像認識入門：[PyTorchを用いたCNNによる画像認識入門](https://qiita.com/momoyama/items/aecf831fbf887f067d42)（Qiita・Conv2d/MaxPool2d/Linear の組み合わせを初心者向けに解説）

## 前回まとめ（指導者が5分）

指導者が第1回（data）の要点を手短にまとめる（受講者にはやらせない）：

- `SPEECHCOMMANDS` → 可変長波形を `pad_or_trim` で1秒に揃える → `collate_fn` で stack → log-mel → `(B,1,n_mels,T')`
- **今日はこの `(B,1,n_mels,T')` を受け取って 35 クラスの logits `(B,35)` を返すモデルを作る**．パイプライン流れ図の2番目の箱．

## 予習発表（15分）

- 第1回の宿題②で分担した model の予習スライドを，代表が1本だけ発表（一人ずつの個別発表はしない）．
- 予習の観点（第1回の宿題②で指定済みのはず）：
  - `nn.Module` を継承するとは何ができるということか（`__init__` と `forward` の役割）
  - `nn.Sequential` は何をするものか
  - `nn.Conv2d` のパラメータ（`in_channels`・`out_channels`・`kernel_size`・`padding`）は何を意味するか
  - `nn.BatchNorm2d` は何をする層か（ざっくり：平均0・分散1に正規化して学習を安定させる）
  - `nn.Linear` の入出力は何か
  - `model.parameters()` は何を返すか
- 指導者は誤りをフォローし，**「いま発表にあった `Conv2d` や `Linear` を実際に組み立てて動かす」**と橋渡しする．

## 当日の流れ（90分目安・タイムボックス）

| 枠 | 内容 | 時間 |
|---|---|---|
| 前回まとめ | data の要点（指導者5分） | 5分 |
| 予習発表 | nn.Module / CNN（受講者が分担した1スライドを代表が1本） | 15分 |
| ハンズオン①：nn.Module 基礎 | 最小例 TinyNet ＋ `model.to(device)` | 10分 |
| ハンズオン②：ConvBlock ＋ AudioCNN | 抽象例→受講者が書く（当日の山場） | 30分 |
| ハンズオン③：shape 追跡 ＋ sanity check | 各層の出力 shape ＋ パラメータ数 ＋ 1バッチ過学習 ＋ check_model PASS | 20分 |
| まとめ | 宿題（model.py 移植＋第3回予習）の指示 | 10分 |

※ ②の ConvBlock ＋ AudioCNN が当日の山場（合計2クラスを書く）．ConvBlock が書けると AudioCNN はそれを並べるだけなので，**ConvBlock に十分時間をかける**．
※ ②が当日中に終わらない人は**宿題で続きをやればよい**と口頭でも明示する（session1 と同じ）．
※ ③が20分に収まらない場合は，base 比較（L226-236）を速い人向けに回し，1バッチ過学習と check_model に集中する．

## ハンズオン①：nn.Module 基礎（10分・動く例）

notebook で**動く最小例**を見せながら，受講者も手元で打って確認する（nn.Module の書き方＝答えでないので動く例でよい）．

### nn.Module の最小例

```python
class TinyNet(nn.Module):
    def __init__(self):
        super().__init__()            # 必須：親の初期化
        self.fc = nn.Linear(28, 10)   # 層を定義

    def forward(self, x):
        return self.fc(x)             # 順伝播を書く

net = TinyNet()
print(net)
print("out:", net(torch.randn(4, 28)).shape)  # (4, 10)
```

- **`__init__`**：使う層を `self.xxx = nn.XXX(...)` で**属性として定義**する．ここで定義した層のパラメータが `model.parameters()` で取れる．
- **`forward`**：入力を層に通す**順伝播の流れ**を書く．`model(x)` と呼ぶと内部で `forward(x)` が走る．
- **`nn.Sequential`**：複数の層を順番に通すだけなら `nn.Sequential(層1, 層2, ...)` でまとめられる（forward を書かずに済む）．
- **層を定義するには入力の shape を知る必要がある**：`nn.Linear(28, 10)` の `28` は入力の次元数．`nn.Conv2d(1, 32, ...)` の `1` は入力のチャネル数．流れてくるデータの形が分からないと層は定義できない．

### model.to(device)（第1回からの橋渡し）

- 第1回で Tensor を `.to('cuda')` して GPU に載せた．**モデルも同じ**：`model.to(device)` でモデルの全パラメータを GPU に移す．
- **入力とモデルは同じ device に揃える**（揃えないとエラー）．
  ```python
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  model = TinyNet().to(device)
  x = torch.randn(4, 28, device=device)   # 入力も同じ device
  print(model(x).shape)
  ```
- 第1回では「Tensor 同士を同じ device に」だったが，今日からは「**model と入力を同じ device に**」が加わる．以降全ての回で使う作法．
- `model.to(device)` の前後で `!nvidia-smi` を打ち，VRAM 使用量が増えることを確認してもよい（第1回の Tensor と同じ感覚）．

## ハンズオン②：ConvBlock ＋ AudioCNN を書く（30分・抽象例→受講者が書く）

ここが当日の**手を動かす中心**．notebook には**抽象例**だけ置き，受講者が中身を書いて動かす．

### ConvBlock（15分）

畳み込みの1ブロック：`Conv2d(3×3)` → `BatchNorm2d` → `ReLU` → `MaxPool2d(2)`．

- **各層の役割**を先に一言ずつ説明してから書かせる（予習発表で触れているはず）：
  - `Conv2d`：画像（2D特徴マップ）に対する畳み込み．`kernel_size=3, padding=1` で空間サイズを保つ．`bias=False` は BN と組み合わせるときの定番（BN が bias 相当を持つため）．
  - `BatchNorm2d`：チャネルごとに正規化．学習を安定させる．
  - `ReLU`：活性化関数（ゼロから作るでやった `max(0, x)`）．
  - `MaxPool2d(2)`：2×2 の窓で最大値を取り，**空間サイズを半分**にする（特徴を圧縮）．

> 【資料作成者向け：配布する抽象例（そのままでは動かない）】
> ```python
> class ConvBlock(nn.Module):
>     def __init__(self, in_ch: int, out_ch: int):
>         super().__init__()
>         # Conv2d(3x3, padding=1, bias=False) → BatchNorm2d → ReLU → MaxPool2d(2)
>         self.block = nn.Sequential(
>             ...
>         )
>
>     def forward(self, x):
>         return self.block(x)
> ```
> 【模範実装（指導者が手元に持つ・配布しない）】
> ```python
> class ConvBlock(nn.Module):
>     def __init__(self, in_ch: int, out_ch: int):
>         super().__init__()
>         self.block = nn.Sequential(
>             nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
>             nn.BatchNorm2d(out_ch),
>             nn.ReLU(),
>             nn.MaxPool2d(2),
>         )
>
>     def forward(self, x):
>         return self.block(x)
> ```

- 書けたら動作確認（out_ch=16 は確認用の任意の値．本番の AudioCNN では base=32 から始まる）：`ConvBlock(1, 16)(torch.randn(2, 1, 64, 101)).shape` → `(2, 16, 32, 50)`（MaxPool で H,W が半減）．

### AudioCNN（15分）

ConvBlock を3段重ね＋ GAP ＋ Linear で分類器を作る．

- Pooling には global average pooling を使う．空間方向を1点に潰して固定長ベクトルにする．
- チャネル幅は `base` 起点：`1 → base → base*2 → base*4`．
- `forward` の最後に `self.classifier(x)` で logits を返すだけ（1行の TODO）．

> 【資料作成者向け：配布する抽象例（そのままでは動かない）】
> ```python
> class AudioCNN(nn.Module):
>     def __init__(self, n_classes: int = 35, base: int = 32):
>         super().__init__()
>         # ConvBlock を 1→base, base→base*2, base*2→base*4 の3段
>         self.features = nn.Sequential(
>             ...
>         )
>         self.gap = nn.AdaptiveAvgPool2d(1)
>         self.classifier = nn.Linear(base * 4, n_classes)
>
>     def forward(self, x):
>         x = self.features(x)            # (B, C, h, w)
>         x = self.gap(x).flatten(1)      # (B, C)
>         # 分類器に通して logits を返す
>         ...
> ```
> 【模範実装（指導者が手元に持つ・配布しない）】
> ```python
> class AudioCNN(nn.Module):
>     def __init__(self, n_classes: int = 35, base: int = 32):
>         super().__init__()
>         self.features = nn.Sequential(
>             ConvBlock(1, base),
>             ConvBlock(base, base * 2),
>             ConvBlock(base * 2, base * 4),
>         )
>         self.gap = nn.AdaptiveAvgPool2d(1)
>         self.classifier = nn.Linear(base * 4, n_classes)
>
>     def forward(self, x):
>         x = self.features(x)
>         x = self.gap(x).flatten(1)
>         return self.classifier(x)
> ```

- 書けたら動作確認：`AudioCNN()(torch.randn(8, 1, 64, 101)).shape` → `(8, 35)`．

### よくあるエラー早見表（②用に配る）

| 症状 | よくある原因 |
|---|---|
| `RuntimeError: mat1 and mat2 shapes cannot be multiplied` | `Linear` の入力次元が合っていない（GAP 後の channel 数 ≠ `Linear` の `in_features`） |
| `expected input[B,X,...] channels, but got Y channels` | ConvBlock のチャネル数の繋がりが間違っている（前段の `out_ch` ≠ 次段の `in_ch`） |
| `Input type (torch.cuda.FloatTensor) and weight type (torch.FloatTensor)...` | model と入力の device が揃っていない（`model.to(device)` か `x.to(device)` を忘れ） |
| `forward` で `NotImplementedError` | `return self.classifier(x)` を書き忘れている（`raise` 文が残っている） |

## ハンズオン③：shape 追跡 ＋ sanity check（20分・動く例）

②で書いた ConvBlock と AudioCNN を**動かして確認する**フェーズ．ここは動く例で見せながら受講者も打って確認．

### shape 追跡（base=32, n_mels=64, T'=101）

各 ConvBlock で `MaxPool2d(2)` が空間を半分にすることを，数値で確認する：

```
入力:                (B,  1,  64, 101)
ConvBlock(1, 32):    (B, 32,  32,  50)    ← MaxPool で H,W 半減
ConvBlock(32, 64):   (B, 64,  16,  25)
ConvBlock(64, 128):  (B, 128,  8,  12)
GAP:                 (B, 128,  1,   1)
flatten:             (B, 128)
Linear(128, 35):     (B, 35)              ← logits
```

- notebook で1段ずつ `model.features[0](x).shape` → `model.features[1](...).shape` → … と追いかけるセルを用意する（動く例）．
- **T'=101 が 50→25→12 になる**：`MaxPool2d(kernel_size=2)` は既定で `stride=2` なので出力は `floor(入力/2)`．奇数の 101 は `floor(101/2)=50` になり，以降偶数なので綺麗に半減．T'=16000//160+1=101（`n_fft=400, hop_length=160, center=True`）．最終的に GAP で 1 になるので端数は問題にならない．
- `sum(p.numel() for p in model.parameters() if p.requires_grad)` でパラメータ数を確認する．

### base を変えるとどうなるか（動く例・観察）

- `base=16` と `base=64` でパラメータ数がどう変わるか試す：
  ```python
  for b in [16, 32, 64]:
      m = AudioCNN(base=b)
      n = sum(p.numel() for p in m.parameters() if p.requires_grad)
      print(f"base={b:2d}: {n:>8,} params")
  ```
- **パラメータ数はチャネル幅（base）の2乗に比例する**（Conv の重みは `in_ch × out_ch × k × k`）．base を2倍にするとパラメータ数は約4倍．
- 「base を小さくすればパラメータ数は減るが精度も下がる．大きくすれば過学習しやすくなる」→ **第4回のチューニングの布石**（容量を増やすとどうなるか？）．

### 1バッチ過学習 sanity check（動く例）

大きなモデルでは1エポックに何時間もかかる．全部回した後にバグに気づいたり，1エポック終わって評価でコケたりすると大きな手戻りになる．**1バッチだけなら数秒で回せる**ので，先に試して実装が正しいか確認しておく．

**同じ1バッチを何百 step も学習して loss が ~0 に落ちるか**を見る．落ちれば forward の実装が正しく，勾配もちゃんと流れている証拠．Karpathy "A Recipe for Training Neural Networks" の定番チェック．

- **初期 loss**：ランダム重みの model は各クラスをほぼ等確率で予測する → `CrossEntropyLoss` の初期値は `-log(1/35) = ln(35) ≈ 3.56`．これより大きく外れていたら何かがおかしい．
- **`CrossEntropyLoss` は生の logits を渡す**（内部で log_softmax ＋ NLL をやるので，**forward 内で softmax をかけてはいけない**＝二重 softmax で勾配が潰れ，学習がうまく進まなくなる）．loss の詳しい話は第3回で．
- 学習ループ（`zero_grad`→`forward`→`loss`→`backward`→`step`）は**ここでは動くコードを提供する**（受講者がループ自体を書くのは第3回の穴埋め）．今はこの5行の中身を完全に理解しなくてよい．loss が下がることだけ確認できれば OK：
  ```python
  model = AudioCNN(n_classes=35, base=32)
  criterion = nn.CrossEntropyLoss()
  optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

  dummy_feats = torch.randn(8, 1, 64, 101)
  dummy_labels = torch.randint(0, 35, (8,))

  model.train()
  for step in range(200):
      optimizer.zero_grad()
      logits = model(dummy_feats)
      loss = criterion(logits, dummy_labels)
      loss.backward()
      optimizer.step()
      if step % 50 == 0 or step == 199:
          print(f"step {step:3d}: loss {loss.item():.4f}")
  ```
- **loss が 3.56 付近から始まって ~0 に落ちれば OK**．落ちなければ forward のどこかで勾配が途切れているか，実装にバグがある．
- 1バッチ過学習に使うデータはランダムテンソルで十分（実装の検証が目的）．`check_model` はランダムテンソルだけで動くので data に依存しない．

### check_model（PASS で model 完成）

当日は notebook のセルとして実行する（`model.py` への移植は宿題）．

宿題で `model.py` に移植した後は，確認スクリプトが `from kws.model import AudioCNN, ConvBlock` で受講者の実装を import して検証する形にする．当日は notebook セル内で直接 `AudioCNN` を参照する．

```python
"""model モジュールの確認：出力 shape・初期 loss・1バッチ過学習・grad 存在を assert する．"""
import sys, math
import torch
from torch import nn

sys.path.insert(0, "src")
from kws.model import AudioCNN, ConvBlock
from kws.utils import set_seed

set_seed(42)

N_MELS, T_PRIME, N_CLASSES, B = 64, 101, 35, 8

# --- 1. 出力 shape ---
model = AudioCNN(n_classes=N_CLASSES, base=32)
dummy = torch.randn(B, 1, N_MELS, T_PRIME)
logits = model(dummy)
assert logits.shape == (B, N_CLASSES), f"出力 shape が (B,35) でない: {logits.shape}"

# --- 2. 初期 loss ≈ ln(35) ≈ 3.56 ---
criterion = nn.CrossEntropyLoss()
targets = torch.randint(0, N_CLASSES, (B,))
init_loss = criterion(logits, targets).item()
expected = math.log(N_CLASSES)  # ≈ 3.555
assert abs(init_loss - expected) < 0.5, f"初期 loss {init_loss:.3f} が ln(35)≈{expected:.2f} から離れすぎ"

# --- 3. backward 後に grad が存在 ---
loss = criterion(model(dummy), targets)
loss.backward()
for name, p in model.named_parameters():
    assert p.grad is not None, f"{name} の grad が None（勾配が流れていない）"

# --- 4. 1バッチ過学習（200 step で loss → ~0） ---
set_seed(42)
model2 = AudioCNN(n_classes=N_CLASSES, base=32)
opt = torch.optim.Adam(model2.parameters(), lr=1e-3)
feats = torch.randn(B, 1, N_MELS, T_PRIME)
labels = torch.randint(0, N_CLASSES, (B,))
model2.train()
for _ in range(200):
    opt.zero_grad()
    out = model2(feats)
    l = criterion(out, labels)
    l.backward()
    opt.step()
final_loss = l.item()
assert final_loss < 0.05, f"200 step 後の loss {final_loss:.4f} が下がりきらない（forward にバグ？）"

print(f"check_model PASS: logits {tuple(logits.shape)}, init_loss {init_loss:.2f}, overfit_loss {final_loss:.4f}")
```

### PASS の見え方

```
check_model PASS: logits (8, 35), init_loss 3.55, overfit_loss 0.0012
```

- `logits.shape == (8, 35)`（B=8, 35クラス）．初期 loss ≈ 3.55（`ln(35)` 付近）．200 step 後の loss < 0.05．
- これが出れば model 完成．**model は data に依存しない**（ランダムテンソルだけで検証が通る）．

## 宿題（次回＝第3回への引き継ぎ）

- **① 実装を `model.py` に移植して整理**：当日 notebook で書いた `ConvBlock`・`AudioCNN` を `src/kws/model.py` の TODO に埋め，`check_model` を PASS させる．
  - `model.py` のスケルトンは配布済み（TODO を埋めるだけ）．`from kws.model import AudioCNN` が通ることを確認．
  - 整理したら「実装の続き＋`check_model` の PASS 結果＋学んだこと」をまとめる．
- **② 第3回（学習ループ）の予習を分担で1スライド**：観点は第3回（`session3_train.md`）の確定・コミット後に指定する．
- 各自「分かったこと1点 ＋ 疑問1点」を持ち寄る．

## まとめで話すこと

- **今日の到達確認**：`check_model` が PASS したか．model はランダム入力でも検証できた＝**data に依存しない単体テスト**ができるのが `nn.Module` の良い所．
- **import の動機づけ（session1 と同じ流れ）**：今日 notebook に書いた ConvBlock と AudioCNN を，宿題で `src/kws/model.py` に移植する．`from kws.model import AudioCNN` で呼べるようにすると，第3回の学習ループからすっきり使える．
- **次回への橋渡し**：今日 model が出す `(B,35)` の logits を，次回は `CrossEntropyLoss` に渡して学習を回す．今日 sanity check で見た `zero_grad → forward → loss → backward → step` が，次回のメインテーマ．

## 次回への引き継ぎ（受け渡しの型）

- model が出す `(B,35)` logits と，data が出す label index `(B,)` を `CrossEntropyLoss` に渡す＝第3回の入口．
- 今日は `model.to(device)` で GPU に載せた．第3回では data の `.to(device)` と合わせて学習ループ内で使う．
- 今日 sanity check で見た学習ループ5行が，第3回で受講者が自分で書く対象になる．

## 速い人向け（任意）

- **base を変えてパラメータ数と初期 loss を比較**：`base=16` / `base=64` で `check_model` を通してみる．パラメータ数が4倍に変わることを確認（チューニングの布石）．
- **ConvBlock を1段増やす**とどうなるか：4段目を足すと GAP 前の空間が `(4, 6)` まで縮む．パラメータ数の変化も確認．
- **わざと device 不一致エラーを起こす**：model を `cuda` に置いて入力を `cpu` のまま渡し，エラーメッセージを読む（実戦でよく遭遇する）．
- **Augmentation を考える**（第1回の布石の続き）：音声データにどんな augmentation が使えそうか調べてみる（SpecAugment・time shift・noise injection 等）．実装は後の回のチューニング向け．

## 資料作成者向けメモ（受講者には出さない）

- 当日は notebook 内で完結（`model.py` 移植は宿題）．`check_model` も当日は notebook セルで実行する設計．session1 と同じパターン．
- `check_model` は **data に依存しない**（ランダムテンソルだけで検証）．data.py の宿題が未完の受講者でも model の作業には支障なし．ただし1バッチ過学習の**実データ版デモ**を見せるなら `from kws.data import get_dataloaders` が必要（指導者が実演するだけなら問題ない）．
- 既存 `notebooks/02_model.ipynb` は旧方針（5回構成・解答公開前提・`git checkout ans` 参照あり）なので，**新方針に合わせてセル構成を刷新する必要がある**．特に Section 7 の `git checkout ans` は削除（解答非公開方針）．
- `train.py` の Config に `base: int = 32` がある．ここで `base` の意味を教えておくことで，第4回チューニングで受講者が自分で変えられる．
- **session1 の宿題②を更新する必要がある**：現在「観点は session2_model.md の確定・コミット後に指定する」となっている箇所に，この文書の「予習発表」節にある問いを入れる．session2 が確定コミットされた時点で更新すること．
- 1バッチ過学習で使う学習ループは第3回の穴埋め対象と重なるが，ここでは「動く例として提供」の位置づけ（受講者にループを書かせるのは第3回）．
- **巡回時のチェックポイント**：受講者が `Conv2d` で `bias=False` を忘れて `bias=True` のまま書いていてもエラーにならず動く．BN が bias 相当を持つため性能にもほぼ影響しない．模範実装と違う点を見つけたら「動くけど慣例的には bias=False にする」と軽く指摘する程度でよい．
