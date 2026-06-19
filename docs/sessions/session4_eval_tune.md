# 第4回 — 学習の深掘り ＋ 評価 ＋ チューニング大会 `session4_eval_tune.md`

> 各回の地図は `docs/sessions/index.md`．本ファイルは第4回の超詳細．
> 対象モジュール：`src/kws/evaluate.py`．完成の確認＝`check_eval` が PASS．

## この回の作り方の方針

- 前半は**学習経過の記録と可視化の導入＋深掘り**．第3回で「ログが残らない」問題を提起した．今回はまず `history.json` への記録と `loss_curve.png` の描画を導入し，その曲線を読んで過学習の分析・early stopping・lr scheduler・seed の話をする．
- 中盤は **evaluate.py の穴埋め**．受講者が書くのは `predict` 関数の中身だけで，confusion matrix の描画と `classification_report` は用意済みのコードを呼ぶだけ．穴は小さく，当日中に終わるようにする．
- 後半は**チューニング大会**．道具が全部揃った状態で，各自がその場で弱ベースライン超えを狙い，leaderboard に申告して締める．時間が余ればチューニングに多く回すが，①②を優先する．
- session1〜3 と同じく**当日は notebook の中だけで完結**させる．`evaluate.py` への移植は宿題．

## ねらい

- 学習経過を `history.json` に記録し，`loss_curve.png` で**学習曲線を可視化する**仕組みを導入する．
- 学習曲線から **overfit / underfit を読み取り**，「次に何をすべきか」の判断ができるようになる．
- early stopping・lr scheduler・seed 固定など，**学習を安定させる道具**を知る．
- `predict` 関数を書いて confusion matrix を出し，**「どのクラスを間違えるか」を分析**できるようになる．
- 達成目標＝**`check_eval` が PASS** ＋ leaderboard に1回以上申告．

## 関連リンク

- `sklearn.metrics.confusion_matrix`：https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html
- `sklearn.metrics.classification_report`：https://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html
- `torch.optim.lr_scheduler`：https://docs.pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate
- 日本語の補助記事：
  - 過学習と正則化：[過学習についての基礎知識](https://qiita.com/FukuharaYohworker/items/26fb33e4a1d4153a5c42) — Qiita．overfit/underfit の図解と対策を網羅
  - Confusion Matrix：[scikit-learnのconfusion_matrixとclassification_reportの使い方](https://qiita.com/ground0state/items/08eb44a4feeb3bb23900) — Qiita．混同行列の読み方と各指標の意味

## 前回まとめ — 指導者が5分

指導者が第3回 train の要点を手短にまとめる．受講者にはやらせない：

- `run_epoch` で学習3ステップ `zero_grad → backward → step` を書いて学習を回した
- print で loss と accuracy を確認して学習が進んでいることを確認した
- ただし**ログが残らない問題**があった．**今日はまずこれを解決する**．その後，曲線を読んで分析し，モデルを評価し，チューニングで精度を上げる．パイプライン流れ図の最後の箱．

## 予習発表 — 15分

- 第3回の宿題②で分担した予習スライドを，代表が1本だけ発表．一人ずつの個別発表はしない．
- 予習の観点は第3回の宿題②で指定済みのはず：
  - 「過学習 overfitting」と「未学習 underfitting」はそれぞれどういう状態か．loss 曲線のどこを見れば分かるか
  - 「early stopping」は何をする仕組みか
  - 「learning rate scheduler」は learning rate を学習中にどう変えるか．なぜ変えるのか
  - 「confusion matrix」は何を表す表か．対角線に数字が集まっているとどういう意味か
  - 「precision / recall / F1」はそれぞれ何を測る指標か
- 指導者は誤りをフォローし，**「いま発表にあった内容を実際にコードで確認してみよう」**と橋渡しする．

## 当日の流れ

| 枠 | 内容 | 時間 |
|---|---|---|
| 前回まとめ | train の要点，指導者5分 | 5分 |
| 予習発表 | 評価・チューニング，代表が1本 | 15分 |
| ハンズオン① | 学習経過の記録・可視化 → 曲線の読み方 → overfit/underfit・early stopping・scheduler・seed | 30分 |
| ハンズオン② | evaluate.py：predict を書く → confusion matrix → classification_report | 20分 |
| チューニング大会 | 各自チューニング → evaluate → leaderboard 申告 | 15分 |
| まとめ | 振り返り・宿題の指示 | 5分 |

※ ①の前半は受講者も手を動かす（history 記録＋plot のコードを書く）．後半は指導者が見せつつ議論する形．
※ チューニング大会は時間が足りなければ宿題に延長してよい．①②を優先する．

## ハンズオン① — 学習経過の記録・可視化 ＋ 深掘り・30分

第3回で「print しかしていないのでログが残らない」という問題を提起した．今回はまずこれを解決し，その上で曲線を読む．

### 学習経過の記録（動く例・受講者も手を動かす）

第3回の学習ループに `history` リストを足して，epoch ごとの loss と accuracy を記録する：

```python
import json

history = []
for epoch in range(1, 11):
    tr_loss, tr_acc = run_epoch(model, loaders['train'], criterion, device, optimizer)
    va_loss, va_acc = run_epoch(model, loaders['val'], criterion, device, None)
    print(f"epoch {epoch}: train loss {tr_loss:.3f} acc {tr_acc:.3f} | val loss {va_loss:.3f} acc {va_acc:.3f}")
    history.append({'epoch': epoch, 'train_loss': tr_loss, 'train_acc': tr_acc, 'val_loss': va_loss, 'val_acc': va_acc})

with open('history.json', 'w') as f:
    json.dump(history, f, indent=2)
```

- `history.json` にまとめて書き出す．学習が途中で止まっても経過が残る．

### loss 曲線の描画（動く例・受講者も手を動かす）

```python
import matplotlib.pyplot as plt

fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))
ep = [h['epoch'] for h in history]
a1.plot(ep, [h['train_loss'] for h in history], 'o-', label='train')
a1.plot(ep, [h['val_loss'] for h in history], 's-', label='val')
a1.set_xlabel('epoch'); a1.set_ylabel('loss'); a1.legend(); a1.grid(alpha=0.3)
a2.plot(ep, [h['train_acc'] for h in history], 'o-', label='train')
a2.plot(ep, [h['val_acc'] for h in history], 's-', label='val')
a2.set_xlabel('epoch'); a2.set_ylabel('accuracy'); a2.legend(); a2.grid(alpha=0.3)
fig.tight_layout(); fig.savefig('loss_curve.png', dpi=150); plt.show()
```

- これで「第3回の問題＝ログが残らない」は解決した．学習曲線の監視には `tensorboard` や `wandb` といったツールもあるが，今回は `history.json` + `loss_curve.png` で十分．

### 学習曲線の読み方（指導者が見せつつ議論）

描いた `loss_curve.png` を全員で読む：

- **train loss↓ / val loss も↓**：まだ学習が進んでいる．epoch を増やす余地がある
- **train loss↓ / val loss↑**：**過学習**．モデルが訓練データに特化しすぎている
- **train loss が高止まり**：**未学習**．モデルの容量が足りないか，学習率が合っていない

10 epoch でも train loss と val loss の乖離が見え始めるはず．「train は良くなっているのに val は良くならない＝過学習が始まっている」ことを確認する．

### early stopping

- **val loss が改善しなくなったら学習を止める**仕組み．patience（何 epoch 待つか）を決めて使う．
- `train.py` は `best.pt` と `last.pt` の2種類を保存する設計になっている．`best.pt` は val accuracy が最良だったエポックの重みで，結果的に early stopping と近い効果がある．ただし early stopping は学習自体を途中で止めて計算時間を節約する仕組みであるのに対し，best.pt 方式は最後まで回してから最良の重みだけ採用するので計算コストは変わらない．
- 「なぜ `last.pt` ではなく `best.pt` を推論に使うのか」を一言：過学習が進んだ後の last より，val が最良だった best のほうが汎化性能が高い．

### lr scheduler

- `train.py` にはすでに `scheduler: cosine` のフックがある．`baseline.yaml` の `scheduler: null` を `cosine` に変えるだけで CosineAnnealingLR が有効になる．
- **CosineAnnealingLR** は学習率を余弦カーブで徐々に下げていく．学習の後半ほど小さなステップでパラメータを微調整する．
- 指導者がデモで見せる：事前に用意した `scheduler: null` と `scheduler: cosine` の loss 曲線画像を並べて比較する．当日その場で再学習はしない．

### seed と再現性

- `set_seed(42)` で乱数を固定しているので，同じ設定で回せば同じ結果が再現する．
- **seed を変えると結果が変わる**ことも確認する：事前に seed=42 と seed=123 で回した結果を並べて，val accuracy が微妙に違うことを見せる．
- チューニング大会で「本当に改善したのか，乱数の揺れなのか」を考えるきっかけにする．

## ハンズオン② — evaluate.py：predict を書く・20分・抽象例→受講者が書く

ここが受講者が手を動かすパート．穴は `predict` 関数の中身だけで，小さく終わるようにする．

### predict 関数の構造

```python
@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    ys, ps = [], []
    for feats, targets in loader:
        feats = feats.to(device)
        logits = model(feats)
        # ここが今日書く部分
        ...
    return torch.cat(ys).numpy(), torch.cat(ps).numpy()
```

- `@torch.no_grad()` で勾配計算を無効にしている．第3回の `torch.set_grad_enabled(False)` と同じ意味．
- `model.eval()` で BatchNorm 等を評価モードにしている．
- やるべきことは「各バッチの予測クラスと正解ラベルを集める」だけ．

### 受講者が書く部分

> 【資料作成者向け：配布する抽象例．そのままでは動かない】
> ```python
> # 予測クラス = logits の argmax
> # 正解ラベル = targets
> # それぞれリストに追加する
> ...
> ```
> 【模範実装．指導者が手元に持つ．配布しない】
> ```python
> ps.append(logits.argmax(1).cpu())
> ys.append(targets.cpu())
> ```

- **`logits.argmax(1)`**：`logits` は `(batch, 35)` の shape で，axis=1 がクラス方向．`argmax(1)` でサンプルごとに最も確率が高いクラスの index を取る．`.cpu()` で CPU に戻す（最後に `.numpy()` で NumPy 配列にするために必要．GPU 上のテンソルは直接 NumPy に変換できない）．
- **`targets`**：正解ラベルも `.cpu()` を明示しておくと device を気にしなくてよい．
- 第3回の `run_epoch` では `logits.argmax(1) == targets` でその場の accuracy だけ計算した．今回は**後で confusion matrix を作るために全件を保持する**必要があるので，リストに append して蓄積する設計にしている．

### 書けたら動作確認

```python
sys.path.insert(0, "../src")
from kws.data import LABELS, get_dataloaders
from kws.model import AudioCNN
from kws.utils import get_device

device = get_device()
ckpt = torch.load('exp/baseline/best.pt', map_location=device, weights_only=False)
base = ckpt.get('config', {}).get('base', 32)
model = AudioCNN(n_classes=35, base=base).to(device)
model.load_state_dict(ckpt['model'])

loaders = get_dataloaders('data', batch_size=256, n_mels=64, num_workers=4)
y_true, y_pred = predict(model, loaders['test'], device)
print(f"test samples: {len(y_true)}, accuracy: {(y_true == y_pred).mean():.3f}")
```

### confusion matrix と classification_report

`predict` が動いたら，用意済みのコードを呼ぶだけ：

```python
from sklearn.metrics import confusion_matrix, classification_report

cm = confusion_matrix(y_true, y_pred, labels=range(35))
print(f"confusion matrix shape: {cm.shape}")  # (35, 35) になるはず

print(classification_report(y_true, y_pred, target_names=LABELS, digits=3))
```

- **confusion matrix** は 35×35 の表で，`cm[i][j]` は「正解が i なのに j と予測した件数」を表す．対角線に数字が集中しているほど正しい予測が多い．
- **classification_report** はクラスごとの precision / recall / F1 を一覧する．precision は「そのクラスだと予測したうち実際に正解だった割合」，recall は「そのクラスの正解のうち見つけられた割合」．全体の accuracy だけでは見えない「特定のクラスだけ極端に弱い」といった問題が分かる．
- 「どのクラスを間違えるか」を実際に読む：混同しやすいペアを見つけて，なぜ間違えるのか考えてみる．

### confusion matrix を画像に保存

```python
from kws.evaluate import plot_confusion
plot_confusion(cm, float((y_true == y_pred).mean()), Path('confusion_matrix.png'))
```

`plot_confusion` は用意済みなので呼ぶだけ．ヒートマップで視覚的に確認する．

### check_eval — PASS で評価パイプライン完成

当日は notebook のセルとして実行する．`evaluate.py` への移植は宿題．

```python
"""evaluate モジュールの確認：predict が動く・出力サイズが正しい・confusion matrix が 35×35 を assert する．"""
import sys
import torch
import numpy as np

sys.path.insert(0, "src")
from kws.data import LABELS, get_dataloaders
from kws.model import AudioCNN
from kws.utils import get_device

from kws.utils import set_seed
set_seed(42)

device = get_device()
ckpt = torch.load('exp/baseline/best.pt', map_location=device, weights_only=False)
model = AudioCNN(n_classes=35, base=ckpt.get('config', {}).get('base', 32)).to(device)
model.load_state_dict(ckpt['model'])

loaders = get_dataloaders('data', batch_size=256, n_mels=64, num_workers=4)
y_true, y_pred = predict(model, loaders['test'], device)

# --- 1. 出力長が test 件数と一致するか ---
n_test = sum(len(t) for _, t in loaders['test'])
assert len(y_true) == n_test, f"y_true の長さ {len(y_true)} が test 件数 {n_test} と一致しない"
assert len(y_pred) == n_test, f"y_pred の長さ {len(y_pred)} が test 件数 {n_test} と一致しない"

# --- 2. confusion matrix が 35×35 か ---
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_true, y_pred, labels=range(35))
assert cm.shape == (35, 35), f"confusion matrix の shape が {cm.shape}，35×35 であるべき"

# --- 3. test accuracy が出るか ---
acc = float((y_true == y_pred).mean())
assert acc > 0.10, f"test accuracy {acc:.3f} がチャンスレートに近すぎる"

print(f"check_eval PASS: test_acc {acc:.3f}, samples {n_test}, cm shape {cm.shape}")
```

### PASS の見え方

```
check_eval PASS: test_acc 0.847, samples 4890, cm shape (35, 35)
```

## チューニング大会 — 15分

道具が全部揃ったので，各自その場でチューニングして弱ベースライン超えを狙う．

### ルール

- **指標は test accuracy**．各自 `predict` → `(y_true == y_pred).mean()` で算出する
- **test split は torchaudio 公式で全員共通固定**．全員が同じテストセットで評価するので比較可能
- 弄ってよいもの：学習率・epoch 数・モデル幅 `base`・`scheduler`・`augment`・モデル構造の変更・その他何でも
- **Slack の #kws-leaderboard チャンネルに申告**する：名前・test accuracy・何を変えたかの一言メモ
- test リークは教育目的で許容する．本来は val で選び test は最後に1回だけ使うべきだが，今回は簡易版ということで許容する．その旨は口頭で説明する
- **seed cherry-picking の禁止**：seed だけ変えて最良の結果を申告するのは NG．変更点を明記する

### チューニングのヒント

指導者が最初に口頭で方向性を示す：

- **まず容量を増やす**：`base=32` → `base=64` にするだけで精度がかなり上がるはず．弱ベースラインは意図的に小さくしてある．**最初に試すべき変更**
- **epoch を増やす**：`epochs=25` → `epochs=50` 等．`best.pt` 採用なので過学習しても最良の重みが残る
- **学習率を変える**：`lr=1e-3` は万能ではない．`lr=3e-4` や `lr=5e-4` を試す
- **scheduler を有効にする**：`scheduler: cosine` に変えてみる
- **モデル構造を変える**：ConvBlock の段数を増やす・Dropout を入れる・残差接続を入れる等．自由
- **Data Augmentation**：時間方向のシフト・ノイズ付加・SpecAugment 等

### 時間の目安

GPU 環境で `base=32` / 25 epoch の学習は約5分．`base=64` にすると約10分．15分のチューニング枠では1回のトライが現実的なので，**まず `base=64` を試す**のがおすすめ．続きは宿題で．

### 大会の進め方

1. 各自 `configs/baseline.yaml` をコピーして設定を変え，`train.py` で再学習する
2. 学習が終わったら `predict` → accuracy を出す
3. Slack に申告する
4. 時間内に複数回トライしてよい．最高記録を申告する
5. 当日中に終わらなくても**宿題として続けてよい**

## 宿題

- **① `evaluate.py` に移植して整理**：当日 notebook で書いた `predict` を `src/kws/evaluate.py` の TODO に埋め，`check_eval` を PASS させる．
  - 移植後は CLI で `uv run python -m kws.evaluate --ckpt exp/baseline/best.pt --device cuda` を実行して，`confusion_matrix.png` と `test_metrics.json` が出力されることを確認する．
- **② チューニングを続ける**：当日中に終わらなかった人はチューニングを続けて，Slack に最高記録を申告する．
- 整理したら「`check_eval` の PASS 結果＋チューニングで試したこと＋学んだこと」をまとめる．

## まとめで話すこと

- **leaderboard の確認**：Slack の申告を確認し，全体の結果を共有する．
- **今日の到達確認**：`check_eval` が PASS したか．confusion matrix を読めたか．leaderboard に申告したか．
- **全4回の振り返り**：data → model → train → eval と積み上げて，「PyTorch で学習・評価スクリプトを自分で書ける」という大目標に到達した．
- **この先への橋渡し**：今回の題材は音声 KWS だったが，同じパイプラインが画像分類・テキスト分類・その他のタスクにも使える．研究で新しいタスクに取り組むときも，data → model → train → eval の流れは変わらない．

## 資料作成者向けメモ — 受講者には出さない

- 当日は notebook 内で完結．`evaluate.py` 移植は宿題．`check_eval` も当日は notebook セルで実行する設計．
- `check_eval` は **best.pt が存在することが前提**．第3回の宿題で `train.py` を回していない受講者がいる場合は，指導者が学習済み best.pt を Slack か共有ドライブで配布する．
- `predict` の穴埋めは実質2行で，15分もあれば終わる．②の時間を短めに設定してチューニング大会に時間を回す．
- leaderboard は Slack チャンネルへの自己申告で十分．スプレッドシート等にまとめるかは当日の空気で判断する．
- test リークは教育目的で許容するが，「本来は test で選んではいけない」ことは**必ず口頭で説明する**．
- 弱ベースラインの test accuracy は 80〜85% 程度を想定．base を 64 にするだけで 90% 前後に届くはず．
- `plot_confusion` は evaluate.py に用意済みなので notebook から import して呼べる．ただし notebook で先に `predict` を定義してからでないと evaluate.py 全体の import は NotImplementedError で止まるので，notebook 内では `from kws.evaluate import plot_confusion` とする．
