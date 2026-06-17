# 第1回 — キックオフ ＋ PyTorch 基礎 ＋ data（`session1_kickoff_data.md`）

> 各回の地図は `docs/sessions/index.md`．本ファイルは第1回の超詳細（読めば誰でも同じノートブックを起こせるレベルを目指す）．
> 対象モジュール：`src/kws/data.py`．完成の確認＝`check_data` が PASS．

## この回の作り方の方針（重要）

- **当日は notebook の中だけで完結させる**：受講者は notebook 内に自分でコードを書いて動かし，`check_data` も notebook のセルで実行する．`src/kws/data.py` への整理（移植）と `import kws` のための editable 設定は**宿題**に回す（当日 import で詰まらせない）．
- **notebook に「クリックすれば動く完成セル」は置かない**：
  - **答えになる実装**（`collate_fn`・`pad_or_trim` など）は **抽象例（関数の骨組み＋処理の流れをコメントで示す・`...` のプレースホルダ）**に留め，受講者が中身を自分で書く．
  - **答えにならない部分**（概念確認・ライブラリ API の呼び出し＝データを開いて見る等）は **動く最小例**を見せてよい（足場として残す）．
- これにより「写経で終わる」「実質解答配布」を避けつつ，実装未経験でも取り組める足場を残す．

## ねらい

- PyTorch の Tensor の基本（`shape`・`dtype`・`device`，`squeeze`/`unsqueeze`）を手で触り，`.to(device)` の意味を体感する．
- SPEECHCOMMANDS のデータを**目で見て**構造を掴む（1サンプルの中身・波形・log-mel）．
- 可変長の波形を**1秒に揃えてバッチに束ねる**（`collate_fn`）流れを理解し，`DataLoader` から `(B,1,n_mels,T')` のミニバッチが出る所まで作る．
- 達成目標＝**`check_data` が PASS**（1バッチが規定の shape / dtype / ラベル範囲で出る）．

## 関連リンク

- PyTorch "Learn the Basics"：Tensors ／ Datasets & DataLoaders（事前課題の予習範囲・当日の土台）
  - 英語：https://docs.pytorch.org/tutorials/beginner/basics/intro.html
  - 日本語訳：https://yutaroogawa.github.io/pytorch_tutorials_jp/
- torchaudio `SPEECHCOMMANDS`：https://docs.pytorch.org/audio/stable/generated/torchaudio.datasets.SPEECHCOMMANDS.html
- torchaudio `MelSpectrogram` / `AmplitudeToDB`：https://docs.pytorch.org/audio/stable/transforms.html

## 当日の流れ（90分目安・タイムボックス）

| 枠 | 内容 | 時間 |
|---|---|---|
| 導入 | 事前課題の動作確認（スモーク PASS）＋ 完成形画像で全体像を共有 | 10分 |
| 予習発表 | Tensors / Datasets&DataLoaders（受講者が分担した1スライドを代表が1本） | 15分 |
| ハンズオン①：PyTorch 基礎 | `tensor`/`shape`/`dtype`/`device`，`squeeze`/`unsqueeze`，`.to('cuda')`＋`nvidia-smi` 実演 | 15分 |
| ハンズオン②：data を見る（動く例を見せる） | 1サンプル構造・波形 plot・log-mel imshow・`label↔index` | 15分 |
| ハンズオン③：data を束ねる（抽象例→受講者が書く） | `pad_or_trim`・`collate_fn` を自分で書く → `DataLoader` で1バッチ → `check_data` PASS | 25分 |
| まとめ | import の動機づけ・宿題（`data.py` 移植＋editable＋第2回予習）の指示 | 10分 |

※ index L119 の警告どおり項目が多い．**クラス分布の確認は宿題に送る**（当日②は最短ルート）．③の `collate_fn`→`check_data` が当日の山場．`batch_size` 体験・Augmentation は速い人向けに置く．
※ ③が当日中に終わらない人は**宿題で続きをやればよい**と口頭でも明示する（「動く所まで」を全員に強要して時間崩壊させない）．

## 導入（10分）

- **事前課題の動作確認**：各自 `uv run python scripts/smoke.py` を走らせ，PASS（version 表示・`SPEECHCOMMANDS: found`）を確認．詰まっている人はここで拾う（当日中に環境を揃える）．
- **完成形の出力例を見せる**：弱ベースラインの画像（confusion matrix・log-mel imshow・loss 曲線）を貼り，**事前課題のパイプライン流れ図と対応づけて**「今日はこの図のいちばん左（data）を作る」と位置づける．※学習済みモデルをその場で動かす実演はしない（解答配布になるため）．

## 予習発表（15分）

- 事前課題で分担した Tensors / Datasets&DataLoaders の予習スライドを，代表が1本だけ発表（一人ずつの個別発表はしない）．
- 指導者は誤りをその場でフォローし，**この後の基礎ハンズオンへの橋渡し**にする（「いま発表にあった `shape`/`device` を実際に触る」）．

## ハンズオン①：PyTorch 基礎（15分）

notebook で**動く最小例**を見せながら，受講者も手元で打って確認する（ここは概念確認なので動く例でよい）．

- **Tensor を作る・形を見る**：`x = torch.randn(2, 3)` → `x.shape`・`x.dtype`（`float32`）．
- **dtype の違い**：`torch.tensor([1,2,3]).dtype`（`int64`）と float の違い．後で**ラベルは整数 `long`**になる伏線．`torch.long` は `torch.int64` の別名（同じ型）と一言．
- **`squeeze`/`unsqueeze`（numpy の復習として）**：`np.squeeze`/`np.expand_dims` と同じ＝**サイズ1の次元を消す/足す**．`torch.randn(1, 16000).squeeze(0).shape`（→`(16000,)`）／`.unsqueeze(1)` で次元を足す．**③で `waveform.squeeze(0)`・log-mel の `unsqueeze(1)` を使う布石**．
- **device**：`x.device`（最初は `cpu`）．`xg = x.to('cuda')` → `xg.device`．
  - **`.to('cuda')` の前後で `nvidia-smi` を打ち，VRAM 使用量が増えるのを実演**（Tensor が GPU に載った＝device の体感）．大きめの Tensor（例 `torch.randn(4096, 4096, device='cuda')`）にすると分かりやすい．
  - **同じ device に揃える**：`cpu` の Tensor と `cuda` の Tensor を足すとエラーになることを見せ，「計算する者同士は同じ device」を体に入れる（モデルを載せるのは第2回）．
- **device・データの流れの概念図**を1枚見せる（文章だと掴みにくいので図．生成画像可）．

## ハンズオン②：data を見る（15分・動く例を見せる）

torchaudio の `SPEECHCOMMANDS` を触り，**データを目で見る**（Karpathy の "データを必ず見る" を体現）．ここは API の使い方＝答えでないので**動く例を見せ，受講者も打って確認**する．

- **データセットを開く**：`SPEECHCOMMANDS(root='data', download=False, subset='training')`（事前課題で取得済みなので `download=False`）．
- **1サンプルの構造**：`ds[0]` が `(waveform, sample_rate, label, speaker_id, utterance_number)` の5要素タプルであることを確認．
  - `waveform.shape` ＝ `(1, T)`（チャンネル1・可変長 T）．`sample_rate` ＝ 16000．`label` ＝ 文字列（例 `"yes"`）．
- **波形を見る**：`matplotlib` で波形を plot．
- **log-mel を見る**：`make_logmel()`（`MelSpectrogram(n_fft=400, hop_length=160, n_mels=64)` → `AmplitudeToDB()`）を1サンプルに適用し，`imshow` で log-mel スペクトログラムを表示．**これがモデルへの入力**だと示す．
  - **このセル（log-mel 変換）と次の `label_to_index` は③でそのまま使う**ので，受講者の手元に残す（②→③の橋渡し）．
- **`label↔index` マップ**：35クラスの `LABELS`（固定順）から `label_to_index` / `index_to_label` を作る．**ラベルは index の整数**（one-hot ではない）＝第3回 `CrossEntropyLoss` に渡る形，と一言．
- ※ **クラス分布の確認は宿題**（分布の棒グラフを宿題で作らせる）．当日は時間を③に回す．

## ハンズオン③：data を束ねる（25分・抽象例 → 受講者が書く）

ここが当日の**手を動かす中心**．notebook には**抽象例（骨組み＋流れのコメント＋`...`）**だけ置き，受講者が中身を書いて動かす．②で手元に残した log-mel 変換・`label_to_index` を使う．

書く対象：

1. **`pad_or_trim(waveform, num_samples=16000)`**：1次元波形を 16000 ちょうどに揃える．
   - 短ければ末尾を 0 パディング，**長ければ先頭から `num_samples` を取る**（`waveform[:num_samples]`．切り詰め位置は全員これで固定＝leaderboard の公平性のため）．
   - **なぜ揃えるか**＝バッチを1つの `(B,T)` テンソルに stack するには全サンプル長が同じである必要があるため（可変長のままでは束ねられない）．
2. **`collate_fn` 内の処理**：各サンプルを1秒に揃えて貯め，label を index 化して貯め，stack → log-mel → `(B,1,n_mels,T')`．

> 【資料作成者向け：配布する抽象例 と 模範実装の切り分け】
> **配布 notebook に置く抽象例（そのままでは動かない）**：
> ```python
> def pad_or_trim(waveform, num_samples=16000):
>     n = waveform.shape[-1]
>     # n < num_samples: 末尾を 0 パディング
>     # n > num_samples: 先頭から num_samples を取る（waveform[:num_samples]）
>     ...
>
> def collate(batch):
>     waveforms, targets = [], []
>     for waveform, _sr, label, *_ in batch:
>         # waveform は (1, T)．1次元にして1秒へ → 貯める
>         # label を index に変換 → 貯める
>         ...
>     # waveforms を stack → (B, T)
>     # log-mel 変換 → channel 次元を足して (B, 1, n_mels, T')
>     # targets を long の Tensor に
>     ...
> ```
> **模範実装（指導者が手元に持つ・配布しない）**：
> ```python
> def pad_or_trim(waveform, num_samples=16000):
>     n = waveform.shape[-1]
>     if n < num_samples:
>         return torch.nn.functional.pad(waveform, (0, num_samples - n))
>     return waveform[..., :num_samples]
>
> def collate(batch):
>     waveforms, targets = [], []
>     for waveform, _sr, label, *_ in batch:
>         waveforms.append(pad_or_trim(waveform.squeeze(0)))
>         targets.append(label_to_index[label])
>     x = torch.stack(waveforms)         # (B, T)
>     feats = transform(x).unsqueeze(1)  # log-mel (B,n_mels,T') → (B,1,n_mels,T')
>     y = torch.tensor(targets, dtype=torch.long)
>     return feats, y
> ```
> つまずき定番：① `waveform` は `(1,T)` なので `squeeze(0)` で1次元に．② log-mel 後に `unsqueeze(1)` で channel を足す．③ ラベルは `dtype=torch.long`．

- 書けたら `DataLoader(ds, batch_size=8, collate_fn=collate, num_workers=0)` で1バッチ取り出して shape を print．
  - **`num_workers=0` で動かす**（notebook/Jupyter で `num_workers>0` はマルチプロセス起因のハング・例外の定番地雷．当日は 0）．
- **詰まり対応**：中間チェックポイントを刻む（`pad_or_trim` 単体で `(16000,)` が返るか → `collate` で1バッチの shape → `check_data`）．下の「よくあるエラー早見表」を配る．

### よくあるエラー早見表（③用に配る）

| 症状 | よくある原因 |
|---|---|
| `stack expects each tensor to be equal size` | `pad_or_trim` で長さが 16000 に揃っていない |
| shape が `(B, T)` のまま / `(B,1,n_mels,T')` にならない | log-mel 変換や `unsqueeze(1)` を忘れている |
| `expected scalar type Long` 系 | ラベルを `dtype=torch.long` にしていない |
| notebook が固まる | `num_workers` を 0 にする |

## 確認スクリプト `check_data`（PASS で data 完成）

当日は notebook のセルとして実行する（`data.py` への移植は宿題）．宿題で `import kws.data` から呼べるようにする．

```python
"""data モジュールの確認：1バッチが規定の形・型・範囲で出るか assert する．
   torch.long は torch.int64 の別名（同じ型）．"""
import torch

N_MELS, NUM_SAMPLES = 64, 16000

# pad_or_trim 単体（前処理仕様の固定＝受講者間で挙動を揃える）
assert pad_or_trim(torch.zeros(5000)).shape[-1] == NUM_SAMPLES      # 短い→パディング
assert pad_or_trim(torch.zeros(30000)).shape[-1] == NUM_SAMPLES     # 長い→切り詰め

# 1バッチを取り出して検証
loader = DataLoader(ds, batch_size=8, collate_fn=collate, num_workers=0)
feats, labels = next(iter(loader))

assert feats.shape == (8, 1, N_MELS, 101), feats.shape   # (B,1,n_mels,T')．T'=16000//160+1=101
assert feats.dtype == torch.float32, feats.dtype
assert labels.dtype == torch.long, labels.dtype          # == torch.int64
assert int(labels.min()) >= 0 and int(labels.max()) <= 34, (labels.min(), labels.max())
assert not torch.isnan(feats).any() and not torch.isinf(feats).any()

print("check_data PASS:", tuple(feats.shape), labels.dtype)
```

### PASS の見え方

```
check_data PASS: (8, 1, 64, 101) torch.int64
```

- `feats.shape == (8, 1, 64, 101)`（B=8, n_mels=64, T'=101）．`labels.dtype == torch.int64`（＝`long`）で値は 0–34．
- これが出れば data モジュール完成．`T'=101` は `n_fft=400, hop_length=160, center=True` から `16000//160+1`．**`n_fft`/`hop_length`/`n_mels`/`LABELS` の順序は変えない**（後の回の入力 shape・leaderboard の前提）．

## 宿題（次回＝第2回への引き継ぎ）

- **① 実装を `data.py` に移植して整理**：当日 notebook で書いたコードを `src/kws/data.py` にまとめ，`check_data` を PASS させる．
  - **`import kws` で使えるようにする（editable 設定）**：事前課題で作った `pyproject.toml` に，`src/kws` をパッケージとして認識させる設定を追記する．これで `from kws.data import ...` が通り，`check_data` が `data.py` 版で動く．
    > 【資料作成者向け】追記する `pyproject.toml` の雛形（hatchling 例）を配布資料に載せる：
    > ```toml
    > [build-system]
    > requires = ["hatchling"]
    > build-backend = "hatchling.build"
    >
    > [tool.hatch.build.targets.wheel]
    > packages = ["src/kws"]
    > ```
    > 追記後 `uv sync`（または `uv pip install -e .`）で editable install される．
  - **クラス分布の確認**：train/val/test の件数と35クラスの分布を棒グラフにする（当日②から送った分）．
  - 整理したら「実装の続き＋`check_data` の PASS 結果＋学んだこと」をまとめる．
- **② 第2回（model）の予習を分担で1スライド**：観点は第2回（`session2_model.md`）の確定後に指定する（`nn.Module` の書き方など）．
- 各自「分かったこと1点 ＋ 疑問1点」を持ち寄る．

## まとめで話すこと

- **import の動機づけ**（①から移動）：今日は notebook に全部書いたが，**コードをファイルに分けて `import` で集めると見通しが良い**．だから宿題で今日のコードを `src/kws/data.py` に移植する，と接続する（ファイル分割の必然性を，実際に切り出す直前のこの位置で語る）．

## 次回への引き継ぎ（受け渡しの型）

- data が出す `(B,1,n_mels,T')` が，第2回 model の入力になる．`label↔index` の整数ラベルが第3回 `CrossEntropyLoss` に渡る．
- 第1回で `.to(device)` を触ったので，第2回で `model.to(device)` に橋渡しする．

## 速い人向け（任意）

- `batch_size` を変えて出力 shape の先頭・メモリの変化を観察する（※ `val`/`test` は `drop_last=False` なので最終バッチだけ B が小さくなる点に注意）．
- **Augmentation に言及だけ**：音声は時間シフト・ノイズ付加などで水増しできる，という発想があること（実装は後の布石・チューニングのネタ）．
- `n_mels` や `hop_length` を変えると log-mel の `imshow` がどう変わるか見る．**ただし提出版 `data.py` には定数を戻す**（後の回・leaderboard の前提を崩さない）．

## 資料作成者向けメモ（受講者には出さない）

- 当日は notebook 内で完結（`data.py` 移植・editable は宿題）．`check_data` も当日は notebook セルで実行する設計．
- `check_data` の `T'=101` は `n_fft=400, hop_length=160, center=True(default)` で `16000//160+1`．`hop_length` を変えると T' が変わるので，資料の数値は config と一致させること．
- 弱ベースラインの数値（val/train）確定後，導入で見せる完成形画像をそれに差し替える．
