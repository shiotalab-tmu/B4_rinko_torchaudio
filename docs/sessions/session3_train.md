# 第3回 — 学習ループ（`session3_train.md`）

> 各回の地図は `docs/sessions/index.md`．本ファイルは第3回の超詳細．
> 対象モジュール：`src/kws/train.py`．完成の確認＝`check_train` が PASS．

## この回の作り方の方針

- session1・session2 と同じく**当日は notebook の中だけで完結**させる．`src/kws/train.py` への移植は**宿題**．
- **答えになる実装**（`run_epoch` 内の学習3ステップ `zero_grad → backward → step`）は**抽象例**（骨組み＋流れコメント＋`...`）で配り，受講者が中身を書く．
- **答えにならない部分**（`CrossEntropyLoss` の挙動・`model.eval()` / `torch.no_grad()` の意味）は**動く例**を見せて受講者も打って確認．
- この回は**最小構成**に絞る：固定エポック・`Adam`・accuracy のみ．best/last 切替・early stopping・lr scheduler・confusion matrix は第4回に回す．

## ねらい

- 学習ループの5行（`zero_grad` → `forward` → `loss` → `backward` → `step`）を**自分の手で書けるようになる**．
- `CrossEntropyLoss` が「生の logits を受け取り，内部で log_softmax + NLL を行う」ことを理解する．
- train / val の分け方と，**`model.eval()` + `torch.no_grad()`** で評価する作法を身につける．
- 学習の経過を **print で追い**，loss が下がっていることを確認する．ログの記録・可視化は第4回で導入する．
- 達成目標＝**`check_train` が PASS**（train loss が下がる・val acc がチャンスレート超え）．

## 関連リンク

- PyTorch "Optimizing Model Parameters"：https://docs.pytorch.org/tutorials/beginner/basics/optimization_tutorial.html
  - 日本語訳：https://yutaroogawa.github.io/pytorch_tutorials_jp/ の同じ章
- `nn.CrossEntropyLoss`：https://docs.pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html
- `torch.optim.Adam`：https://docs.pytorch.org/docs/stable/generated/torch.optim.Adam.html
- 日本語の補助記事：
  - 学習ループ：[Pytorchの基礎 forwardとbackwardを理解する](https://zenn.dev/hirayuki/articles/bbc0eec8cd816c183408)（forward/backward の仕組みと勾配計算の流れを図解）
  - 勾配計算：[PyTorch の勾配とか誤差逆伝播が何をやっているのか世界一詳しく説明する](https://qiita.com/hibit/items/f32930dcf3d8ac5889cc)（zero_grad → backward → step の各ステップを内部動作レベルで詳述）

## 前回まとめ（指導者が5分）

指導者が第2回（model）の要点を手短にまとめる（受講者にはやらせない）：

- `ConvBlock`（Conv2d → BN → ReLU → MaxPool）を3段積んだ `AudioCNN` で `(B,1,64,101)` → `(B,35)` logits
- 1バッチ過学習で実装の正しさを確認した．**今日はこの logits と正解ラベルを使って実際に学習を回す**．パイプライン流れ図の3番目の箱．

## 予習発表（15分）

- 第2回の宿題②で分担した train の予習スライドを，代表が1本だけ発表（一人ずつの個別発表はしない）．
- 予習の観点（第2回の宿題②で指定済みのはず）：
  - 「損失関数」は何のためにあるか．`nn.CrossEntropyLoss` はどういう場面で使うか
  - 「optimizer」は何をするものか．`optimizer.step()` を呼ぶと何が起きるか
  - 「学習率」は何を決めるパラメータか
  - `loss.backward()` は何を計算するか（「勾配」がパラメータにどう紐づくか）
  - 学習ループの5行（`zero_grad` → `forward` → `loss` → `backward` → `step`）はそれぞれ何をしているか
- 指導者は誤りをフォローし，**「いま発表にあった学習ループを実際に自分で書いて動かす」**と橋渡しする．

## 当日の流れ

| 枠 | 内容 | 時間 |
|---|---|---|
| 前回まとめ | model の要点（指導者5分） | 5分 |
| 予習発表 | 学習ループ（受講者が分担した1スライドを代表が1本） | 15分 |
| ハンズオン①：CrossEntropyLoss を理解する | loss の挙動を動く例で確認 | 10分 |
| ハンズオン②：学習ループを書く | `run_epoch` の学習3ステップを受講者が書く（当日の山場） | 25分 |
| ハンズオン③：学習を回して確認する | 数エポック回す → print で確認 → test accuracy → check_train | 25分 |
| まとめ | 宿題（train.py 移植＋第4回予習）の指示 | 10分 |

※ ②の学習3ステップが当日の山場．第2回の1バッチ過学習で5行のうち3行は見ているので，受講者は「あれを自分で書く」感覚で入れる．
※ ②が当日中に終わらない人は**宿題で続きをやればよい**と口頭でも明示する．
※ ②が15分を超えても動かない受講者には，指導者が模範実装の該当行だけ個別に見せて先に進ませる．③の check_train PASS まで到達することを優先する．
※ ③が25分に収まらない場合は，print 出力で loss が下がっていることだけ確認して `check_train` に進む．

## ハンズオン①：CrossEntropyLoss を理解する（10分）

notebook で**動く例**を見せながら，受講者も手元で打って確認する．

- **CrossEntropyLoss に渡すもの**：`logits (B, C)` と `labels (B,)`（整数 index・`long`）．第1回で作った `label_to_index` の整数がそのままここに来る．
  ```python
  criterion = nn.CrossEntropyLoss()
  logits = torch.randn(4, 35)          # 4サンプル，35クラス
  labels = torch.tensor([0, 5, 12, 34])  # 正解の index
  loss = criterion(logits, labels)
  print(f"loss: {loss.item():.4f}")
  ```
- **softmax をかけてはいけない**：`CrossEntropyLoss` は内部で `log_softmax` を行う．forward の出力に `softmax` をかけてから渡すと二重になり，勾配が潰れて学習がうまく進まなくなる．完全にエラーにはならないが性能が大幅に劣化する．第2回で一言触れた内容の再確認．
- **初期 loss ≈ ln(35) ≈ 3.56**：ランダム重みなら各クラスをほぼ均等に予測する → 初期 loss は `-log(1/35)` 付近になる．第2回の sanity check で確認済みの数値．
- **loss が小さい＝予測が正解に近い**：loss を最小化するのが学習の目的，と一言．
- **手を動かす確認**：logits の特定クラスを大きい値にしたら loss はどう変わるか，受講者に予想させてから実行させる．「正解クラスの logit を大きくすると loss が下がる」ことを自分の目で確認する．

## ハンズオン②：学習ループを書く（25分・抽象例→受講者が書く）

ここが当日の**手を動かす中心**．notebook には**抽象例**だけ置き，受講者が中身を書いて動かす．

### run_epoch の構造（先に全体像を見せる）

```python
def run_epoch(model, loader, criterion, device, optimizer=None):
    is_train = optimizer is not None
    model.train(is_train)         # 学習モードと評価モードを切り替える
    total, correct, loss_sum = 0, 0, 0.0

    for feats, targets in loader:
        feats, targets = feats.to(device), targets.to(device)
        with torch.set_grad_enabled(is_train):
            logits = model(feats)              # forward
            loss = criterion(logits, targets)  # loss
        if is_train:
            # ここが今日書く部分（学習の3ステップ）
            ...
        loss_sum += loss.item() * targets.size(0)
        correct += (logits.argmax(1) == targets).sum().item()
        total += targets.size(0)

    return loss_sum / total, correct / total
```

- **`optimizer` を渡すと学習モード，渡さないと評価モード**．同じ関数で train / val を兼ねる．
- **`model.train()` vs `model.eval()`**：BatchNorm や Dropout の挙動が変わる（BN は train 時にバッチ統計を更新し，eval 時に蓄積した running_mean / running_var を使う）．これは**モデルの内部動作モード**の切替．
- **`torch.set_grad_enabled(is_train)`**：勾配計算の on/off を切り替える．`with torch.no_grad():` と同じ効果だが，`is_train` の値で切り替えられるので train/val 兼用の関数に向いている．これは**autograd**の話で，上の `train()/eval()` とは別の仕組み．eval 時は**両方**設定する必要がある．

### 受講者が書く部分（学習の3ステップ）

> 【資料作成者向け：配布する抽象例（そのままでは動かない）】
> ```python
> if is_train:
>     # 1. 勾配をリセット
>     # 2. 誤差逆伝播（勾配を計算）
>     # 3. パラメータを更新
>     ...
> ```
> 【模範実装（指導者が手元に持つ・配布しない）】
> ```python
> if is_train:
>     optimizer.zero_grad()
>     loss.backward()
>     optimizer.step()
> ```

- **`zero_grad()`**：前の step の勾配をリセットする．PyTorch は勾配を**足し込む**ので，リセットしないと前の step の勾配が残ってしまう．
- **`loss.backward()`**：計算グラフを辿って各パラメータの勾配を計算する（ゼロから作るの「誤差逆伝播法」がこの1行）．
- **`optimizer.step()`**：計算された勾配を使ってパラメータを更新する（ゼロから作るの「SGD の更新式」がこの1行．今回は Adam）．

### 書けたら動作確認

```python
import sys
sys.path.insert(0, "../src")
from kws.data import get_dataloaders
from kws.model import AudioCNN
from kws.utils import set_seed

set_seed(42)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
loaders = get_dataloaders('data', batch_size=256, n_mels=64, num_workers=4)
model = AudioCNN(n_classes=35, base=32).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# 1 epoch だけ回して loss と accuracy を確認
tr_loss, tr_acc = run_epoch(model, loaders['train'], criterion, device, optimizer)
print(f"train: loss {tr_loss:.3f}, acc {tr_acc:.3f}")
```

- train loss が初期（≈3.56）から下がっていれば学習が回っている証拠．1 epoch で 2.5〜3.0 程度まで下がれば順調．
- `num_workers=4` でデータローディングを並列化する．0 だと学習が非常に遅くなる．

### よくあるエラー早見表（②用に配る）

| 症状 | よくある原因 |
|---|---|
| loss が下がらない（3.56 付近で横ばい） | `zero_grad` / `backward` / `step` のどれかが抜けている |
| loss が `nan` になる | forward で softmax をかけた logits を `CrossEntropyLoss` に渡している（二重 softmax） |
| `RuntimeError: element 0 ... does not require grad` | `model.eval()` のまま学習しようとしている / `torch.no_grad()` 内で backward を呼んでいる |
| GPU メモリ不足 | `batch_size` を小さくする（128 や 64 で試す） |
| `NotImplementedError: run_epoch の学習ステップを...` | TODO を埋める前に実行している．3行を書いてから再実行する |
| val なのに loss が下がり続ける | val 呼び出しで `optimizer=None` を渡し忘れている（optimizer を渡すと学習モードで動いてしまう） |

## ハンズオン③：学習を回して確認する（25分）

②で書いた `run_epoch` を使って**数エポック学習を回し，print 出力で確認**する．

### 学習ループ（固定エポック・最小構成）

②の動作確認で1 epoch 分学習が進んだ model をそのまま使い続ける．epoch の通し番号も続きから始めてよい（②→③のセル実行順序が結果に影響するので，結果の数値は多少ブレてよい）．

```python
for epoch in range(1, 6):  # 5 epoch（当日は時間の都合で少なめ）
    tr_loss, tr_acc = run_epoch(model, loaders['train'], criterion, device, optimizer)
    va_loss, va_acc = run_epoch(model, loaders['val'], criterion, device, None)
    print(f"epoch {epoch}: train loss {tr_loss:.3f} acc {tr_acc:.3f} | val loss {va_loss:.3f} acc {va_acc:.3f}")
```

- **val は `optimizer=None` で呼ぶ**．パラメータを更新しない．
- epoch ごとに train / val の loss と accuracy を print する．
- train loss が epoch ごとに下がっていれば学習は進んでいる．

### ここで気づく問題：ログが残らない

今のループは print しかしていないので，後から振り返れない：

- 途中で notebook を閉じたら，どこまで進んだか分からない
- 学習率や epoch 数を変えて比較したくても，前回の結果が消えている
- 長時間の学習を走らせて後から結果を確認したい場合に困る

→ **この問題を第4回で解決する**．学習経過をファイルに記録し，loss 曲線を描いて可視化する仕組みを導入する．

### test accuracy

学習が終わったら test で1回だけ評価して accuracy を出す（confusion matrix 等は第4回）：

```python
te_loss, te_acc = run_epoch(model, loaders['test'], criterion, device, None)
print(f"test accuracy: {te_acc:.3f}")
```

- 弱ベースラインで当日5 epoch だと val 30〜40% 程度．宿題で 15〜25 epoch 回すと val 60〜70% 付近まで伸びる．

### check_train（PASS で学習ループ完成）

当日は notebook のセルとして実行する（`train.py` への移植は宿題）．

宿題で `train.py` に移植した後は，確認スクリプトが `from kws.train import run_epoch` 等で受講者の実装を import して検証する形にする．

```python
"""train モジュールの確認：学習が回る・loss が下がる・成果物が出る を assert する．"""
import sys, json
import torch
from torch import nn

sys.path.insert(0, "src")
from kws.data import get_dataloaders
from kws.model import AudioCNN
from kws.utils import set_seed

set_seed(42)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
loaders = get_dataloaders('data', batch_size=256, n_mels=64, num_workers=4)
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

print(f"check_train PASS: train_loss {losses[0]:.3f}→{losses[-1]:.3f}, val_acc {va_acc:.3f}, test_acc {te_acc:.3f}")
```

### PASS の見え方

```
check_train PASS: train_loss 3.401→2.156, val_acc 0.312, test_acc 0.298
```

- train loss が下がっている（学習が回っている）．val/test accuracy がチャンスレート ≈2.9% をはるかに超えている．
- 3 epoch だけなので精度は低いが，**学習ループが正しく動いている**ことが確認できればよい．宿題で epoch を増やして本格的に学習する．

## 宿題（次回＝第4回への引き継ぎ）

- **① 実装を `train.py` に移植して整理**：当日 notebook で書いた `run_epoch` を `src/kws/train.py` の TODO に埋め，`check_train` を PASS させる．
  - `train.py` のスケルトンは配布済み（TODO を埋めるだけ）．移植後は CLI で `uv run python -m kws.train --config configs/baseline.yaml --device cuda --epochs 15` で本格学習を回す．
  - `exp/baseline/` に `last.pt`・`best.pt` が出力される．この `best.pt` が第4回の入力になる．
  - 整理したら「実装の続き＋`check_train` の PASS 結果＋学んだこと」をまとめる．
- **② 第4回（評価・チューニング）の予習を分担で1スライド**：観点は第4回（`session4_eval_tune.md`）の確定・コミット後に指定する．
- 各自「分かったこと1点 ＋ 疑問1点」を持ち寄る．

## まとめで話すこと

- **今日の到達確認**：`check_train` が PASS したか．学習ループの5行を自分で書けた．
- **宿題の指示**：`train.py` に移植したら，`baseline.yaml` で15 epoch 回して `best.pt` を作っておく．
- **今日の問題提起の振り返り**：今のループは print しかしていないのでログが残らない．次回はまずこの問題を解決する（学習経過の記録と可視化）．その上で曲線を読んで overfit / underfit を判断し，confusion matrix で「どのクラスを間違えるか」を分析する．

## 次回への引き継ぎ（受け渡しの型）

- `train.py` が出力する `best.pt`（val accuracy 最良の重み）が，第4回 `evaluate.py` の入力になる．
- 今日の「ログが残らない」問題を第4回で解決する：学習経過の記録（`history.json`）と可視化（`loss_curve.png`）の導入．
- 今日は `run_epoch` で train/val を回した．第4回では同じ作法で test を回し，confusion matrix を出す．

## 速い人向け（任意）

- **学習率を変えてみる**：`lr=1e-2` と `lr=1e-4` で loss 曲線がどう変わるか比較（学習率の感覚をつかむ）．
- **エポック数を増やす**：10〜25 epoch 回して，過学習の兆候（train loss↓ / val loss↑）が出るか観察する．第4回への布石．
- **tmux でバックグラウンド学習**：`tmux` を使って `uv run python -m kws.train ...` をバックグラウンドで走らせる練習（実務 Tips）．
- **nvidia-smi で GPU 使用率を確認**：学習中に別ターミナルから `nvidia-smi` を打ち，GPU 使用率と VRAM を確認する．

## 資料作成者向けメモ（受講者には出さない）

- 当日は notebook 内で完結（`train.py` 移植は宿題）．`check_train` も当日は notebook セルで実行する設計．
- `check_train` は **data.py と model.py に依存する**（`get_dataloaders` と `AudioCNN` を import）．第1回・第2回の宿題が未完の受講者がいる場合は，指導者が完成版を一時的に提供する必要がある．
- `train.py` のスケルトンには `run_epoch` の5行全体が TODO になっている（`optimizer.zero_grad()` / `loss.backward()` / `optimizer.step()` の3行を埋める）．`forward` と `loss` の行は既に書いてあるので，受講者が書くのは実質3行．
- wandb は第3回では触れない．第4回で可視化の話をする際に「tensorboard や wandb というツールもある」と軽く紹介する．
- baseline.yaml の `epochs: 25` は宿題で使う設定．当日の notebook では 5 epoch 程度に留める．
- **train.py の TODO コメントに模範解答の関数名が書かれている**（`optimizer.zero_grad()` / `loss.backward()` / `optimizer.step()`）．配布版では「1. 勾配リセット 2. 誤差逆伝播 3. パラメータ更新」のようにヒントレベルに薄める（notebook の抽象例と同じ粒度に揃える）．
- 第1回・第2回の宿題が未完の受講者がいる場合，**前回まとめの前に**指導者が完成版の data.py / model.py を Slack で配布する．指導者は**前日までに**完成版ファイルを用意して手元で `check_data` / `check_model` が PASS することを確認しておくこと．
