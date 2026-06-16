> **【DISCONTINUED・旧版】** このドキュメントは過去の検討段階の記録で，**現行方針ではありません**．
> 当時どう考えていたかを残す目的で保管しています．現行の方針は別管理（リポジトリ刷新時に作成予定）．

# 各回の内容まとめ

PyTorch と torchaudio で音声コマンド認識（keyword spotting, KWS）を一通り実装するハンズオン教材の，
各回の解説をまとめたドキュメント．指導者の準備用，および受講者の振り返り用．

## 全体像

- 題材: `torchaudio.datasets.SPEECHCOMMANDS`（1 秒程度の音声コマンド 35 クラス）
- ねらい: 環境構築（uv）→ データローダー → モデル → 学習ループ → 評価 → 実験管理（wandb）を，
  生の PyTorch を手で書いて一通り体験する
- 想定受講者: B4・PyTorch 未経験．深層学習と CNN の理論は履修済み．理論より「PyTorch でどう書くか」に重点
- 進め方: notebook を上から実行し，`# TODO` のセルを埋めて動かす．各回の最後に，埋めたコードを
  `src/kws/` の対応ファイルへ整理する
- 全 5 回．第 1〜4 回で一通りさらい，第 5 回は性能向上のワークショップ

### ブランチ運用
- `main`: 受講者が穴埋めする版．`src/kws/` はスケルトン（要所が `TODO` / `NotImplementedError`），
  notebook は穴埋め式
- `ans`: 完成版．`src/kws/` の完成コードと実行済み notebook を回ごとに 1 コミットずつ収録．
  答え合わせは `git checkout ans -- src/kws/<file>`

### ファイル構成
```
src/kws/
  data.py      データセット・ラベル変換・collate_fn・log-mel 変換
  model.py     AudioCNN（log-mel 入力の 2D-CNN → GAP → Linear）
  train.py     学習ループ（CLI）．config / checkpoint / resume / wandb
  evaluate.py  test 評価（accuracy / 混同行列 / クラス別指標）
  utils.py     set_seed / get_device / checkpoint 入出力 / パラメータ数
configs/baseline.yaml   ハイパーパラメータ
notebooks/01〜05.ipynb  教科書本体
exp/<run_name>/         実験成果物（checkpoint・図・指標・ログ）
```

---

## 第 1 回 — PyTorch 超入門 + データを触る

- notebook: `01_setup_and_data.ipynb` ／ 対象: `src/kws/data.py`
- 達成目標: `DataLoader` を回すと log-mel スペクトログラムのミニバッチ `(B, 1, n_mels, T)` が出てくる

### 扱う内容
1. 環境構築（uv）: `uv init` → `uv add torch torchaudio ...`，または `uv sync`．GPU が見えるか確認
2. Tensor の基礎: `shape` / `dtype` / `device`，`x.to(device)` で同じ device に揃える
3. autograd: `requires_grad=True` → `.backward()` → `.grad` に勾配が自動で入ることを，手計算と一致させて確認
4. SPEECHCOMMANDS: 1 サンプルの構造 `(waveform, sample_rate, label, ...)`，波形とラベル分布の可視化
5. ラベル ↔ index: 35 クラスを固定順で並べ `label_to_index` / `index_to_label` を作る
6. `pad_or_trim`: 可変長の波形を 1 秒（16000 サンプル）に揃える（短ければ 0 埋め，長ければ切り詰め）
7. log-mel 変換: `MelSpectrogram` → `AmplitudeToDB` を `nn.Sequential` で組み，imshow で「画像」を確認
8. `collate_fn`: 「長さ揃え → stack → log-mel → ラベル index 化」を 1 つにまとめ，`DataLoader` に渡す

### 穴埋め対象
`label_to_index` / `index_to_label`，`pad_or_trim` の本体，`make_collate_fn` の中身

### 補足
- train / val / test は SPEECHCOMMANDS が同梱する `validation_list.txt` / `testing_list.txt` で一意に定義される
  （ランダム分割ではない）．`subset="training"` は両リストに含まれない残り全部
- 特徴量変換（log-mel）は前処理側（`data.py`）に置く．モデルは分類器に専念する

---

## 第 2 回 — モデルを作る（nn.Module）

- notebook: `02_model.ipynb` ／ 対象: `src/kws/model.py`
- 達成目標: ランダム入力から 35 次元の logits が出る／1 バッチを過学習できる

### 扱う内容
1. `nn.Module` の書き方: `__init__` で層を属性として定義，`forward` で順伝播を書く（最小例で確認）
2. `ConvBlock`: `Conv2d(3x3) → BatchNorm2d → ReLU → MaxPool2d(2)`．MaxPool で時間・周波数を半分に
3. `AudioCNN`: `ConvBlock` を 4 段（`base → base*2 → base*4 → base*4`）→ Global Average Pooling →
   `Linear(35)`．GAP により入力の時間長に依存しない固定次元になる
4. 入出力 shape とパラメータ数（約 25 万）を確認
5. 損失と最適化: `CrossEntropyLoss`（logits をそのまま渡す）と `Adam` の呼び出し方
6. sanity check: 同じ 1 ミニバッチを数十ステップ学習し loss がほぼ 0 に落ちるか見る．
   落ちれば「forward → loss → backward → step」の配線が正しい証拠

### 穴埋め対象
`ConvBlock` の層定義，`AudioCNN` の `features` と `forward` の出力

---

## 第 3 回 — 学習ループ + wandb

- notebook: `03_train_wandb.ipynb` ／ 対象: `src/kws/train.py`, `configs/baseline.yaml`
- 達成目標: 学習が回り，loss が下がる曲線が出る（matplotlib / wandb）

### 扱う内容
1. 学習ループの 5 行: `zero_grad()` → forward → loss → `backward()` → `step()`．
   勾配は累積するので毎ステップでリセットする．評価時は `torch.no_grad()` で勾配を流さない
2. `run_epoch`: optimizer を渡したら学習，渡さなければ評価，と 1 つの関数で両対応．
   `model.train()` / `model.eval()` の切り替えにも注意
3. train / val を分けて数 epoch 回す（notebook では 2 epoch．本学習は CLI から）
4. checkpoint: `last.pt`（毎エポックの現在の重み＋optimizer/epoch → 中断再開用）と
   `best.pt`（val accuracy が最良のエポック → 推論・配布用）の 2 種類
5. ログの可視化: `history` から matplotlib で loss/acc 曲線を描く → 同じことを wandb（`init` + `log`）で
6. 補足: early stopping と best-checkpoint の違い／混合精度 AMP（深入りはしない）／lr scheduler
   （[研究室まとめ](https://github.com/tenk-9/pytorch_scheduler_list) へリンク）

### 穴埋め対象
`run_epoch` の学習 3 ステップ（zero_grad / backward / step）

### CLI での本学習
```bash
uv run python -m kws.train --config configs/baseline.yaml --device cuda          # exp/baseline/ に成果物
uv run python -m kws.train --config configs/baseline.yaml --device cuda --resume # last.pt から再開
```
`exp/<run_name>/` には `last.pt` / `best.pt` / `history.json` / `loss_curve.png` / `train.log` /
`config.json` が残る．既存の run ディレクトリはタイムスタンプ付きで退避してから上書きする．

---

## 第 4 回 — 評価と再現性

- notebook: `04_eval_repro.ipynb` ／ 対象: `src/kws/evaluate.py`, `exp/baseline/`
- 達成目標: 学習済みモデルの test accuracy と混同行列を出せる

### 扱う内容
1. checkpoint の読み込み: `best.pt` から重みと学習設定（`config`）を取り出し，同じ形のモデルに流し込む
2. `predict`: `@torch.no_grad()` で test セットを推論し，正解と予測を集める（予測は logits の argmax）
3. accuracy とクラス別レポート（`classification_report`: precision / recall / F1）
4. 混同行列: 行＝正解・列＝予測．対角から外れた濃いマスが混同しているクラス対
5. 混同しやすいクラス対を非対角成分の大きい順に確認（似た音が並ぶ）
6. 再現性: `set_seed` で乱数固定，config と `uv.lock` を残す，run ディレクトリ規約

### 穴埋め対象
`predict` の予測・正解の収集

### 成果物
評価で `confusion_matrix.png` / `confusion_matrix.npy` / `test_metrics.json`（accuracy とクラス別 P/R/F1 を全部保存）を出力

---

## 第 5 回 — 性能向上ワークショップ

- notebook: `05_workshop.ipynb`
- 達成目標: 自分の工夫でベースライン（test acc ≈ 0.92）を超える

### 扱う内容
- 改善施策のメニュー: data augmentation（SpecAugment ほか）／lr scheduler／label smoothing／
  モデル幅・深さ／学習量／正則化／early stopping
- SpecAugment（時間・周波数マスク）の効果を図で確認
- lr scheduler の考え方と CosineAnnealing の挙動（[研究室まとめ](https://github.com/tenk-9/pytorch_scheduler_list)）
- `configs/*.yaml` を増やして `--config` で実験を切り替え，`--run-name` で結果を分ける
- wandb で複数 run を重ねて比較
- 仮説 → 実験 → 比較 → 考察のサイクル．1 度に 1 施策ずつ変えるのがコツ

---

## ベースラインの実績

- 設定: epochs 25 / batch 256 / Adam lr 1e-3 / log-mel 64 mel / AudioCNN（base 32, 約 25 万パラメータ）/ seed 42
- 結果: val best ≈ 0.92（epoch 24）, **test accuracy ≈ 0.92 / macro F1 ≈ 0.91**
- epochs を 25 と長めにとり，train accuracy が 0.99 まで上がる一方で val が頭打ちになる
  「過学習」の様子をあえて見せている．best.pt を採用するので test 精度は落ちず，第 5 回の改善余地になる

## 環境・再現

- パッケージ管理は uv（`pyproject.toml` / `uv.lock`）．torch / torchaudio は CUDA(cu121) ホイール
- データ（約 2.3GB）と `exp/` の中身は git 管理外（各自ダウンロード・生成）
- 学習・評価は GPU マシンで実行する
