# 事前課題 — 環境構築 ＋ 第1回の予習

## やること

- **全員**：環境構築 + SPEECHCOMMANDS データセットの取得 → 確認スクリプトを PASS させる
- **分担**：Tensors / Datasets & DataLoaders の予習 → 1スライドにまとめる

## 詰まったとき

- まず 30 分は自分で粘る．無理なら **Slack で相談**（実行したコマンド ＋ **エラーメッセージ全文**を貼ると解決が速い）
- メンバー同士の相談OK．指導者が時間外もサポートする
- どうしても進まなくても，当日一緒に直せるので，まずは手を動かすこと

## 関連リンク

- uv 公式：https://docs.astral.sh/uv/
- uv で PyTorch を入れる：https://docs.astral.sh/uv/guides/integration/pytorch/
- PyTorch "Learn the Basics"：https://docs.pytorch.org/tutorials/beginner/basics/intro.html
  - 読む章＝**Tensors** と **Datasets & DataLoaders**
- PyTorch チュートリアル日本語訳：https://yutaroogawa.github.io/pytorch_tutorials_jp/
- torchaudio `SPEECHCOMMANDS`：https://docs.pytorch.org/audio/stable/generated/torchaudio.datasets.SPEECHCOMMANDS.html
- 参考書：『ゼロから作る Deep Learning』①
  - shape / dtype → 1巻 1章．バッチ → 1巻 4章

## 手順

ターミナルで進める．

### 1. clone して中に入る

```bash
git clone <リポジトリのURL>
cd <リポジトリ名>
```

### 2. uv をインストール

```bash
uv --version
```

バージョンが出ればインストール済み．`command not found` なら uv 公式の手順で入れる．

### 3. uv プロジェクトを作る

```bash
uv init --python 3.12
```

Python は **3.12** を指定する（torch 2.4.1 は 3.13 に対応していない）．

### 4. 依存パッケージを足す

```bash
uv add "torch==2.4.1" "torchaudio==2.4.1" matplotlib librosa
```

### 5. データセットを取得

```bash
uv run python -c "import torchaudio; torchaudio.datasets.SPEECHCOMMANDS(root='data', download=True)"
```

約 2.3GB．空き容量 3GB 以上を確保しておく．

### 6. 確認スクリプトを実行

```bash
uv run python scripts/smoke.py
```

`torch` / `torchaudio` の version が表示され，`SPEECHCOMMANDS: found` と出れば環境完成．

GPU が無く `CUDA avail : False` でも PASS 扱い（GPU は研究室マシンで使う）．

## 予習

PyTorch 公式チュートリアル "Learn the Basics" の **Tensors** と **Datasets & DataLoaders** を読み，以下の問いに自分の言葉で答える．

分担の割り当ては参加者でよしなに．担当外もざっと目を通しておくと当日スムーズ．

### (A) Tensors 系

- テンソルを作るとは何をしていることか
- `shape` とは何を表すか
- `dtype` とは何か．`float32` と `int64` は何が違うか
- `device` とは何か
- `.to(device)` は何をする処理か
- 計算する2つのテンソルの `device` が違うとどうなるか
- `squeeze` と `unsqueeze` は何をする操作か

> 第1回では `waveform.squeeze(0)` でチャネル次元を外したり `unsqueeze(1)` でチャネル次元を足したりする場面がある．

### (B) Datasets & DataLoaders 系

- `Dataset` は何の役割を持つクラスか
- `DataLoader` は何の役割を持つクラスか．`Dataset` と何が違うか
- 「バッチ」とは何か．なぜ1件ずつでなく束ねて処理するのか
- `DataLoader` は内部で複数件をどうやって1つのテンソルにまとめているか
- `collate_fn` は何のためにあるか．どういうときにカスタムの `collate_fn` が必要か

> 第1回では音声データの長さがサンプルごとに違うため，カスタム `collate_fn` を書いて長さを揃えてからバッチにまとめる．

## 宿題

- **環境を完成させる**：確認スクリプトが PASS する状態にしておく
- **予習を分担で1スライドにまとめる**：上の問いに自分の言葉で答える．担当を厚めに，担当外は軽くてよい
- 各自「**分かったこと1点 ＋ 疑問1点**」を持ち寄る
