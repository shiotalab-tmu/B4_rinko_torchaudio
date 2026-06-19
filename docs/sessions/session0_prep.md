# 事前課題 — 環境構築 ＋ 第1回の予習

> 各回の概要は `docs/sessions/index.md`．本ファイルは事前課題の説明．
> 事前課題は各自が行う（次週発表）．

## ねらい

- **自分の手で uv プロジェクトをゼロから立て**，torch / torchaudio が動く環境を作れるようになる．
  環境を自分で作れること自体がうれしいので，`uv sync` ではなく `uv init` をやってもらう．
- SPEECHCOMMANDS（35クラス・約2.3GB）を取得し，**スモークスクリプトが PASS** する状態にする．
- これから4回かけて作る **PyTorch の学習パイプライン全体の流れ**を，図で先に掴む．
- PyTorch 基礎（Tensors / Datasets&DataLoaders）を予習し，第1回（data）の予習を**分担で1スライド**にまとめる．

## 事前課題でやること

- **全員がやること**：① 環境構築／② SPEECHCOMMANDS データセットの取得．
  → 下の「手順」に従って進め，最後に**確認スクリプトを PASS**させる．
- **分担してやること**：Tensors / Datasets / DataLoaders の予習（下の「予習」を参照）．
  → どの資料を見て・どんな問いについて・どの粒度でまとめるかは「予習」の節で具体的に示す．分担の割り当ては参加者でよしなに．

## 詰まったときの取り組み方

- まず 30 分は自分で粘る．無理なら **Slack で相談**（実行したコマンド ＋ **エラーメッセージ全文**を貼ると解決が速い）．
- **メンバー同士の相談OK**．指導者が時間外もサポートする．
- 環境構築でどうしても進まなくても，**当日（第1回）に一緒に直せる**ので，まずは手を動かしてみること．

## 関連リンク

- uv 公式（install / `uv init` / `uv add`）：https://docs.astral.sh/uv/
- uv で PyTorch を入れる：https://docs.astral.sh/uv/guides/integration/pytorch/
- PyTorch "Learn the Basics"：https://docs.pytorch.org/tutorials/beginner/basics/intro.html
  - 読む章＝**Tensors** と **Datasets & DataLoaders**．
- PyTorch チュートリアル：https://yutaroogawa.github.io/pytorch_tutorials_jp/
  - 上の公式の日本語訳．英語がつらい人はこちらで同じ章を読む．
- torchaudio `SPEECHCOMMANDS`：https://docs.pytorch.org/audio/stable/generated/torchaudio.datasets.SPEECHCOMMANDS.html
- 参考書：『ゼロから作る Deep Learning』①．
  - shape / dtype の感覚 → 1巻 1章（NumPy の多次元配列）．「バッチ」「ミニバッチ」の感覚 → 1巻 4章．

## 配布物（この時点で渡すもの）

- 教材リポジトリ（notebook・`src/kws/` スケルトン・確認スクリプト・`scripts/smoke.py`）．
  - **配布リポジトリには `pyproject.toml` と `uv.lock` を含めない**（受講者が `uv init` からゼロで作るため．含まれていると `uv init` がエラーで止まる）．
  - `scripts/smoke.py` は **公式 API を呼ぶだけで「答え」を含まない**ので配布してよい．

## まず着地点を見る（全体像を図で掴む）

作業に入る前に，**これから4回かけて何を作るのか**を1枚の図で掴む．これが全4回の地図になる．

> 【資料作成者向け】ここに **PyTorch の学習パイプラインの流れ図**を載せる（生成画像可）．図に含める要素：
> - 流れ：`Dataset`（1サンプル取り出す）→ `DataLoader`（バッチにまとめる・`collate_fn`）→ `Model`（forward）→ `Loss` → `backward()` → `optimizer.step()`．
>   評価のときは同じ `Model` に `no_grad` で通すだけ（`backward` しない）．
> - 各箱を通るときの**テンソル shape の変化**を軽く添える：`waveform` → `(B, 1, n_mels, T')` → `(B, 35) logits` → スカラー（loss）．
> - **どの箱を誰がいつ作るか**の対応：`Dataset`/`DataLoader`＝第1回，`Model`＝第2回，`Loss`/`backward`＝第3回，`no_grad` 評価＝第4回．
> - 図に**含めない**もの（後の回に取っておく）：モデル内部の層構成（第2回）／`device`・`.to()` の話（第1回の概念図に譲る）／実際の関数名・コード．
> - **今回の予習（Tensors / Datasets&DataLoaders）は，この図の左側（`Dataset` → `DataLoader`）に対応する**と明示する．右側（Model 以降）はまだ予習しなくてよい．

この図の流れを4回かけて自分の手で組むと，最終的に「log-mel スペクトログラム・loss 曲線・35×35 の confusion matrix」といった出力が出せるようになる．

## 手順（各自・自宅）

ターミナル（シェル）で進める．基本の Bash コマンドも一緒に確認する．各手順に「うまくいかないとき」を添える．

1. **教材リポジトリを clone して中に入る**（Bash の基本も確認）．
   ```bash
   git clone <リポジトリのURL>
   pwd            # いまどこにいるか（カレントディレクトリ）を表示
   ls             # clone されたフォルダがあるか確認
   cd <リポジトリ名>   # そのフォルダに入る
   pwd            # 入れたか（パスが変わったか）を確認
   ```
   - うまくいかないとき：`ls` でフォルダが見えない → clone が失敗している．URL を確認して再実行．
2. **uv をインストール**．
   - まず**入っているか確認**：
     ```bash
     uv --version
     ```
     - `uv 0.x.x` のように出れば**インストール済み**（次へ）．
     - `uv: command not found` と出たら未インストール → 上の「uv 公式」リンクの手順で入れる．．
3. **uv プロジェクトをまっさらから作る**：リポジトリのルートで
   ```bash
   uv init --python 3.12
   ```
   - 作られるファイルの意味：`pyproject.toml`（依存やプロジェクト設定の定義）／`.python-version`（使う Python のバージョン＝3.12）．`main.py` 等も作られるが今は無視してよい．
   - **Python は 3.12 を指定する**（torch 2.4.1 が対応するのは 3.8〜3.12．3.13 だと入らない）．
   - うまくいかないとき：「`pyproject.toml` が既にある」と出たら，消してから再実行．
4. **依存パッケージを自分で足す**（`pyproject.toml` に追記される）：
   ```bash
   uv add "torch==2.4.1" "torchaudio==2.4.1" matplotlib librosa
   ```
   - 入れるもの：`torch`／`torchaudio`／`matplotlib`／`librosa`．
   - うまくいかないとき：解決が遅い／失敗するときはもう一度実行．
5. **データセット（SPEECHCOMMANDS）を取得**：
   ```bash
   uv run python -c "import torchaudio; torchaudio.datasets.SPEECHCOMMANDS(root='data', download=True)"
   ```
   - `uv run python ...` ＝ いま作った uv 環境の Python でコードを実行する，という意味．
   - これは torchaudio の公式 API を1行呼んで，データをダウンロード・展開している．展開先は `data/SpeechCommands/speech_commands_v0.02/`．
   - 確認：`ls data/SpeechCommands/speech_commands_v0.02/` でフォルダの中身（各コマンドのフォルダ）が見えれば取得成功．
   - 目安：回線次第で数分〜十数分．**空き容量は 3GB 以上**確保．途中で失敗したら再実行でよい．
6. **確認スクリプトを実行**して環境を確認：
   ```bash
   uv run python scripts/smoke.py
   ```
   下の PASS の見え方になれば環境完成．

## 確認スクリプト `scripts/smoke.py`

```python
"""環境チェック：import・version・CUDA・データの有無を確認する．
   データの DL はしない（download を踏ませない）．DL は手順5で明示的に行う．"""
from pathlib import Path
import torch
import torchaudio

print("torch       :", torch.__version__)
print("torchaudio  :", torchaudio.__version__)
print("CUDA avail  :", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device      :", torch.cuda.get_device_name(0))

# データは「既に取得済みか」だけ確認する（自動 DL を踏ませない）
sc_dir = Path("data") / "SpeechCommands" / "speech_commands_v0.02"
if sc_dir.exists():
    print("SPEECHCOMMANDS: found  ->", sc_dir)
else:
    print("SPEECHCOMMANDS: NOT found. 手順5のデータ取得をまだ実行していません．")
```

### PASS の見え方（これが出れば事前準備は OK）

**このスクリプトが下のように通れば，事前準備は完了**（当日はこれの動作確認だけ）：

- **必須**：`torch` / `torchaudio` の version が表示される ／ `SPEECHCOMMANDS: found -> ...` と出る．
- **任意**：`CUDA avail : True`．
  - **自宅 PC に GPU が無く `CUDA avail : False` でも事前課題は PASS 扱い**（GPU は研究室マシンで使う／第1回で実演する）．import とデータ取得が通っていれば完成．

## 予習（PyTorch 基礎・第1回 data の下ごしらえ）

予習は2つの問い群に分かれている．**当日の分担は tenk が割り振る**が，各自**両方ともざっと目を通しておく**と当日スムーズ．

各問いには，**自分の言葉で1〜2文**で説明を書く（できれば図やコード1行などの具体例を1つ添える）．
公式の文章をそのまま貼るのではなく，「自分が人に説明するときの言い方」にする．正解の暗記が目的ではない．

### (A) Tensors 系
"Learn the Basics" の **Tensors** 章（日本語訳でも可）を読みながら，次の問いに答える：

- テンソルを作るとは何をしていることか（`torch.tensor` / `torch.zeros` など）
- `shape` とは何を表すか
- `dtype` とは何か．`float32` と `int64`（`long`）は何が違うか
- `device` とは何か（CPU と GPU でテンソルが「どこにあるか」の話）
- `.to(device)` は何をする処理か
- 計算する2つのテンソルの `device` が違うとどうなるか
- `squeeze` と `unsqueeze` は何をする操作か（NumPy の `squeeze` / `expand_dims` と同じ）

> 補助：shape / dtype の感覚は『ゼロから作る Deep Learning』①の 1章（NumPy の多次元配列）が足がかりになる．
> 第1回では `waveform.squeeze(0)` でチャネル次元を外したり `unsqueeze(1)` でチャネル次元を足したりする場面がある．

### (B) Datasets & DataLoaders 系
"Learn the Basics" の **Datasets & DataLoaders** 章を読みながら，次の問いに答える：

- `Dataset` は何の役割を持つクラスか（何を受け取り，何を返すか）
- `DataLoader` は何の役割を持つクラスか．`Dataset` と何が違うか
- 「バッチ」とは何か．なぜ1件ずつでなく束ねて処理するのか
- `DataLoader` は内部で複数件をどうやって1つのテンソルにまとめているか
- `collate_fn` は何のためにあるか．どういうときにカスタムの `collate_fn` が必要か

> 補助：「バッチ／ミニバッチ」の感覚は『ゼロから作る Deep Learning』①の 4章（学習）が足がかりになる．
> 第1回では音声データの長さがサンプルごとに違うため，カスタム `collate_fn` を書いて長さを揃えてからバッチにまとめる．

## 宿題（次回＝第1回への引き継ぎ）

- **① 環境を完成させる**：スモークが PASS する状態にしておく．
- **② 予習を分担で1スライドにまとめる**：上の (A)(B) の問いについて，自分の言葉でまとめる．
  - **分担の割り当ては当日 tenk が行う**（初回なので説明の担当はこちらで分ける）．便宜上 (A) Tensors 系・(B) Datasets&DataLoaders 系の2群．
  - 担当外の問いまで厚く書かなくてよい．自分の担当を厚めに．
  - 当日は代表が1枚に統合して 10〜15 分で1本だけ発表（一人ずつの個別発表はしない）．
  - **発表や院試で忙しい人は予習を軽くしてよい**．
- 各自「**分かったこと1点 ＋ 疑問1点**」を持ち寄る．

## 第1回への受け渡しメモ（資料作成者向け・受講者には出さない）

- 事前課題では `src/kws` を `import kws` で使う設定は**入れない**（スモークは単体スクリプトなので不要）．
- 第1回以降の宿題（`data.py` / `model.py` への移植）では，確認スクリプト冒頭に `sys.path.insert(0, "src")` を入れて `from kws.data import ...` を通す方式にする（editable install は不要）．
