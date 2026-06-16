> **【DISCONTINUED・旧版】** このドキュメントは過去の検討段階の記録で，**現行方針ではありません**．
> 当時どう考えていたかを残す目的で保管しています．現行の方針は別管理（リポジトリ刷新時に作成予定）．

# PyTorch 音声情報処理チュートリアル 実施計画

## Context（なぜ作るか）
PyTorch を使った深層学習による音声情報処理の「雰囲気」を，**環境構築（uv）→ データローダー作成 →
Model クラス → 学習ループ → 評価 → wandb での実験管理**まで一通り体験してもらうハンズオン教材を作る．
題材は torchaudio の `SPEECHCOMMANDS`（35 クラスのキーワードスポッティング = KWS: Keyword Spotting）．
KWS は「音声から決まった少数の単語だけを検出する」タスク（"OK Google" の wake word や yes/no/up/down の
コマンド認識など）で，本データセットはその標準ベンチマーク．

全5回．第1〜4回で一通りさらい，第5回は「ベースラインからの性能向上ワークショップ」とする．
最終的に受講者の手元に **(1) 各種 python ファイル，(2) 実験結果ディレクトリ，(3) 説明資料（notebook）**
が残り，実験が再現可能であることをゴールとする．

### 受講者像（教材設計の前提）
- **B4・PyTorch 未経験**（PyTorch のプログラムを書いたことがない規模感）
- 深層学習・CNN の**理論は履修済み**（畳み込み・損失・勾配降下は既知）．足りないのは PyTorch 実装の経験
- → **理論説明は最小限**にし，「PyTorch でどう書くか／全体の流れ」に集中する

### 決定事項（tenk と合意済み）
- **ベースライン**: log-mel スペクトログラム + 小型 2D-CNN（`AudioCNN`）
- **タスク規模**: 全 35 クラス
- **実行環境**: GPU 前提（研究室マシンに SSH）．**device は明示指定**を基本とする
  （`--device cuda` / config の `device:`）．未指定時のみ auto 判定にフォールバック可
- **回数**: 全 5 回．最終回は性能向上ワークショップ
- **環境構築**: uv．**`uv init` → `uv add` をゼロからハンズオンでやる**（第1回の最初に組み込む）
- **教材スタイル**: notebook が本体．**skeleton（TODO）を穴埋めして動かす**雰囲気体験重視．
  各回末尾で，穴埋めしたコードを `src/kws/*.py`（こちらも TODO skeleton）に整理 → ファイルが残る
- **発展トピックは削る/「ちらっと」**: AMP/dtype は1セル「こんなのもある」＋リンク，Hydra は名前だけ，
  特徴量変換の置き場所論は深入りしない（`data.py` に置くと決め打ち）

## 設計方針（notebook で穴埋めして動かす → src に整理）
notebook が**教科書本体**．説明セルを厚めにしつつ，コードは要所を **TODO（穴埋め）** にして受講者が埋める．
「全部ゼロから書く」ではなく「骨組みは与え，肝の数行を自分で埋めて動かす」ことで未経験者でも挫折せず
全体の流れを体験できる．各回末尾の「モジュール化」で，埋めたコードを `src/kws/*.py` の該当 TODO に
移して整理 → 再利用可能なファイルが手元に残る．第3回以降はその py を CLI（`uv run python -m kws.train ...`）
で回し，結果を `exp/` に残す．

**毎回「今日これが動いた」という達成**を持たせる（未経験者のモチベ維持）．各回に達成目標を明示．

## 模範解答の管理（ブランチ戦略）
- **main ブランチ**: 教材本体（notebooks/）＋ `src/kws/` は**スケルトン**
  （関数・クラスのシグネチャ + docstring + `TODO`/`raise NotImplementedError`）．受講者はこれを穴埋めする．
- **ans ブランチ**: 完成版 `src/kws/` を**回ごとに 1 コミット**ずつ積む
  （例: c1 "第1回: data.py", c2 "第2回: model.py", c3 "第3回: train.py" …）．
- 運用: 進捗は ans の `git log` で辿れる．答え合わせは `git checkout ans -- src/kws/data.py` で該当ファイルだけ取得．
  指導者のスモークテストは ans ブランチで実行する．
- README に「main で穴埋め／詰まったら ans を参照」のワークフローを明記．

## プロジェクト構成
```
B4_rinko_torchaudio/
├── pyproject.toml          # uv 管理（完成版）．torch/torchaudio(+CUDA index), wandb, soundfile 等
├── uv.lock                 # 環境を揃えたい人向けの確定版
├── README.md               # 環境構築・進め方・各回への導線・参考ソース
├── plan.md                 # 本計画
├── .gitignore              # .venv/, exp/, data/, wandb/ を除外
├── src/kws/                # main=スケルトン（TODO）/ ans ブランチ=完成版（模範解答）
│   ├── __init__.py
│   ├── data.py             # SPEECHCOMMANDS Dataset, label<->index, collate_fn(1秒固定), log-mel transform
│   ├── model.py            # AudioCNN(nn.Module): log-mel入力の 2D-CNN → GAP → Linear
│   ├── train.py            # 学習ループ(CLI). config/seed/device(明示)/wandb/checkpoint
│   ├── evaluate.py         # test 評価. accuracy, confusion matrix, per-class
│   └── utils.py            # set_seed, get_device(name), save/load checkpoint, count_params
├── configs/
│   └── baseline.yaml       # ハイパラ（lr, batch, epochs, n_mels, model幅, device など）
├── notebooks/              # 教科書本体（TODO 穴埋め式）
│   ├── 01_setup_and_data.ipynb
│   ├── 02_model.ipynb
│   ├── 03_train_wandb.ipynb
│   ├── 04_eval_repro.ipynb
│   └── 05_workshop.ipynb
├── scripts/
│   └── run_baseline.sh     # ベースライン再現用ワンコマンド
└── exp/                    # 実験成果物（checkpoint, figures, metrics.json）．構造はコミット, 中身はignore
```

## 参考ソース（教材の説明粒度・流れの土台にする一次資料）
notebook の説明は，画像など他ドメインでもよいので**信頼できる一次ソース**の流れを踏襲する．
- PyTorch 公式 "Learn the Basics": Datasets & DataLoaders / Build Model / Optimization / Save & Load
  - https://docs.pytorch.org/tutorials/beginner/basics/data_tutorial.html
  - https://docs.pytorch.org/tutorials/beginner/basics/buildmodel_tutorial.html
  - https://docs.pytorch.org/tutorials/beginner/basics/optimization_tutorial.html
- torchaudio 公式 "Speech Command Classification with torchaudio"（本題材の定番）
  - Colab（現行・有効）: https://colab.research.google.com/github/pytorch/tutorials/blob/gh-pages/_downloads/c64f4bad00653411821adcb75aea9015/speech_command_classification_with_torchaudio_tutorial.ipynb
  - .py 版: https://docs.pytorch.org/tutorials/_downloads/2abcc2842c010aaabe1256ec4e23422a/speech_command_classification_with_torchaudio_tutorial.py
  - 注: 旧 `intermediate/...html` は 404（公式 tutorials の intermediate から外れたがソースは上記に残存）
- wandb 公式 PyTorch integration（`wandb.init` / `log` / `watch`）
  - https://docs.wandb.ai/models/integrations/pytorch
- 補助: Dive into Deep Learning（https://d2l.ai/）（CNN / 学習ループ / 正則化の教科書的説明）
- uv 公式（環境構築・PyTorch CUDA index 指定）: https://docs.astral.sh/uv/

### 参照しないが「ネタ元」として活用するもの
- NVIDIA NeMo "Speech_Commands.ipynb"（https://github.com/NVIDIA/NeMo/blob/stable/tutorials/asr/Speech_Commands.ipynb）
  - **準拠はしない**：NeMo + PyTorch Lightning + OmegaConf の高レベル抽象で，学習ループ・Model・データ・実験管理が
    すべて隠蔽され，今回の「生 PyTorch を手で書いて雰囲気をつかむ」目的に逆行する．
  - **拝借する点**：MatchboxNet の軽量アーキ発想，SpecAugment/time-shift の augmentation（→第5回），
    confidence でソートした誤分類分析（→第4回）．

## 想定時間配分（1回 80 分）と事前準備
全5回・各80分で分量は概ね妥当．ラスト約15分を「src への整理」に充て，残り約65分が説明+穴埋めハンズオン．

**事前準備（README に明記し，授業前に各自完了させる）**
- 重い DL を授業時間から外す: `uv sync`（torch+CUDA）と SPEECHCOMMANDS（~2.3GB）の取得を事前に
- wandb アカウント作成と `wandb login`（第3回まで）

**各回の目安**
- 第1回: 環境構築の確認 10 / Tensor・autograd 小導入 15 / データ確認・可視化 20 / collate・shape 15 / 整理 10
- 第2回: nn.Module の書き方 15 / AudioCNN 穴埋め 25 / shape・params 10 / loss・optim 10 / 整理 10
- 第3回（山場・最も丁寧）: 学習ループ 30 / checkpoint 10 / ログ(CSV→wandb) 20 / 整理 10（AMP/dtype は1セル流すだけ）
- 第4回: 評価指標 20 / confusion・誤分類 20 / 再現性(seed) 10 / 整理 10
- 第5回: 施策の紹介 15 / 各自実験 50 / 共有・講評 15

**過密リスクと対策**: 第1回（重 DL）と第3回（学習ループ＋wandb 初回）が詰まりやすい．
→ DL とアカウント作成を事前準備に回す．第3回は学習ループ理解を最優先にし，AMP/dtype は深入りしない．

## 各回の内容と粒度
> 進め方：指導者が notebook を説明 → 受講者が TODO を穴埋めして動かす → 各回ラスト10〜15分で src に整理．
> 理論（CNN・勾配降下など）は履修済み前提で最小限．「PyTorch でどう書くか」と「全体の流れ」に集中．

### 第1回 — PyTorch 超入門 + データを触る
notebook: `01_setup_and_data.ipynb`，対象: `src/kws/data.py`
- **環境構築をハンズオンで**: `uv init` → `uv add torch torchaudio`（CUDA index の指定方法も）
  → `uv add matplotlib soundfile pyyaml wandb` → `uv run python -c "import torch; print(torch.cuda.is_available())"`
- **Tensor / autograd の小導入**（PyTorch 未経験向け）:
  - `torch.tensor`，基本演算，`shape`/`dtype`/`device`，`x.to(device)`（CPU↔GPU と「同じ device に揃える」）
  - `requires_grad=True` → 何か計算 → `.backward()` → `.grad` で「**勾配が自動で出る**」を体感（autograd の気持ち）
- `torchaudio.datasets.SPEECHCOMMANDS(subset=...)` の DL と 1 サンプル構造（waveform, sr, label）
- 波形・ラベル分布の可視化，35 クラスの label↔index マップ
- **形状(shape)を逐一確認する習慣づけ**：各ステップで `tensor.shape` を print して目で追う
- `collate_fn` で可変長 → 1 秒(16000)固定（pad/trim），`Dataset`/`DataLoader` の役割
  - なぜ揃えるか＝**バッチ化**：ミニバッチを 1 つの `(B, T)` テンソルに stack するため（簡潔に）
- `MelSpectrogram` + `AmplitudeToDB` で log-mel 変換（理論はさっくり）．**どんな「画像」が流れるか**を imshow で確認
- 整理: ラベルマップ・`collate_fn`・log-mel transform を `data.py` へ
- **達成目標**: DataLoader を回すと「log-mel のミニバッチ `(B,1,n_mels,T)`」が出てくるのを確認できる

### 第2回 — モデルを作る（nn.Module の書き方に集中）
notebook: `02_model.ipynb`，対象: `src/kws/model.py`
- `nn.Module` の構造（`__init__` で層を定義 / `forward` で繋ぐ）— 理論でなく**書き方**を見せる
- `AudioCNN`（log-mel 入力の 2D-CNN）を**穴埋め**で作る: 入力 (B,1,n_mels,T) → Conv2d-BN-ReLU-Pool × 3〜4
  → GAP → Linear(35)．conv block の TODO を埋める
- 入出力 shape の追跡，パラメータ数（`count_params`）
- 損失（`CrossEntropyLoss`）と optimizer（Adam）の **PyTorch での呼び出し方**（役割は既知前提）
- sanity check: **1 バッチ過学習**（面白い確認として）．1 ミニバッチを数十 step 学習し loss→ほぼ0 を確認 →
  「配線が正しい」ことの保証．行かなければラベルずれ・勾配が流れない等のバグ
- （注）特徴量変換は前回の `data.py` 側に置く（model は分類器に専念）．配置論には深入りしない
- 整理: `AudioCNN` を `model.py` へ
- **達成目標**: ランダム入力 →（35次元 logits が出る）／1 バッチを過学習できる

### 第3回 — 学習ループ + wandb（最重要・最も丁寧に）
notebook: `03_train_wandb.ipynb`，対象: `src/kws/train.py`, `configs/baseline.yaml`
- **学習ループを一行ずつ丁寧に**（ここが山場）: `optimizer.zero_grad()` → `forward` → `loss` →
  `loss.backward()` → `optimizer.step()`．**なぜこの順か**（勾配のリセット→計算→更新）を腹落ちさせる
- train / val を分ける理由，epoch，accuracy の計算，`model.train()`/`model.eval()` と `torch.no_grad()`
- **device 明示**転送（前回の autograd 体感とつなげる）
- checkpoint 保存は **2 種類**：`last.pt`（毎エポックの現在重み＋optimizer/epoch → **中断再開**用）と
  `best.pt`（val acc 最良 → 推論・配布用）．`--resume` で `last.pt` から学習を継続できる．
  **なぜ必要か**＝最終 epoch が最良とは限らない（best）／途中で落ちても再開できる（last+resume）／推論・配布（best）
- **early stopping と best-checkpoint** の違いも解説：baseline は**あえて early stopping せず**最後まで回し
  （best.pt で最良を保持），過学習の様子を見せる＋第5回の改善余地を残す．baseline は **epochs=25**（過学習が見える程度）．
- ログ可視化（二段構え）: まず **print/CSV → matplotlib で描く**（透明）→ 同じことを **wandb に置換**して
  `wandb.init`+`wandb.log` の最小2 API で自動可視化・run 比較を体感．`WANDB_MODE=offline` も紹介
- **（ちらっと）AMP/dtype は1セルだけ**: 「精度には fp32/fp16/bf16 など色々あり，低精度で省メモリ・高速化できる
  仕組み（AMP）もある」と紹介し詳細はリンクへ（深入りしない）
  - 参考: Wikipedia "bfloat16"（https://en.wikipedia.org/wiki/Bfloat16_floating-point_format）/
    Google Cloud "BFloat16"（https://cloud.google.com/blog/products/ai-machine-learning/bfloat16-the-secret-to-high-performance-on-cloud-tpus）/
    PyTorch AMP recipe（https://docs.pytorch.org/tutorials/recipes/recipes/amp_recipe.html）
- config は**素の YAML（pyyaml）＋ dataclass/argparse** で読む（透明・低認知負荷）．Hydra は名前だけ紹介
- notebook で 1〜2 epoch → CLI `uv run python -m kws.train --config configs/baseline.yaml --device cuda` で本学習
- 整理: 学習ループを CLI 化して `train.py` へ
- **達成目標**: 学習が回り，loss が下がる曲線が wandb（or CSV プロット）に出る

### 第4回 — 評価と再現性
notebook: `04_eval_repro.ipynb`，対象: `src/kws/evaluate.py`, `exp/baseline/`
- test 評価: overall accuracy / confusion matrix / per-class accuracy
- 誤分類例の可視化・試聴，混同するクラス（似た音）の考察
- 再現性（軽く）: `set_seed`，cudnn deterministic，config と `uv.lock` の保存，run ディレクトリ規約
- `exp/<run_name>/`（ckpt, metrics.json, figures, config）の整理
- 整理: 評価ロジックを `evaluate.py` へ
- **達成目標**: 学習済みモデルの test accuracy と confusion matrix が出せる

### 第5回 — 性能向上ワークショップ
notebook: `05_workshop.ipynb`
- ベースライン acc を起点に改善施策を各自試し wandb で比較（簡易 leaderboard）
- 施策メニュー: augmentation（time shift / `BackgroundNoise` 加噪 / SpecAugment=`FrequencyMasking`+`TimeMasking`），
  正規化，lr scheduler，label smoothing，モデル幅/深さ，batch/epoch，early stopping
- 変更は `configs/*.yaml` 追加で切替（第3回の最小フックを使用）／ train.py を直接いじってもよい
- 仮説 → 実験 → wandb 比較 → 考察のサイクルを自走
- （発展紹介・名前だけ）多数の実験を一括で回すなら Hydra の multirun/sweep が便利
- **達成目標**: 自分の工夫でベースラインの accuracy を超える

## 主要な技術メモ（実装時の注意）
- **torchaudio バージョン**: `SPEECHCOMMANDS` は内部で `torchaudio.load` を使う．torchaudio 2.11+ は
  `torchcodec` 必須で `ModuleNotFoundError` が出るため，`torchaudio<2.9` をピン（または `soundfile`/`torchcodec` を追加）．
  → `pyproject.toml` で torch/torchaudio の CUDA index（例 cu121）を `[[tool.uv.index]]` で明示．
- **device**: `get_device(name)` で `--device`/config 指定を尊重．指定なしのときだけ `cuda if available else cpu`．
- **データ DL 量**: SPEECHCOMMANDS v0.02 は ~2.3GB．`data/` は gitignore．README に DL 手順を明記．
- **wandb**: アカウント作成 or `WANDB_MODE=offline`．README とプランに両対応を記載．
- **特徴量変換**: log-mel は `data.py`（前処理側）に置き，model は分類器に専念（配置論は教材では深入りしない）．
- **config**: 本筋は素の YAML（pyyaml）＋ dataclass/argparse．AMP/dtype・Hydra は「ちらっと/名前だけ」．

## 検証（どう動作確認するか）
1. 空ディレクトリから `uv init` → `uv add ...`（or 同梱 `pyproject.toml` で `uv sync`）でクリーン構築が通る
2. `01〜04` の notebook を上から Run All してエラーなく通る（穴埋め済みの ans ブランチで検証）
3. `uv run python -m kws.train --config configs/baseline.yaml --device cuda --epochs 1` がスモーク完走し `exp/` に成果物
4. `uv run python -m kws.evaluate --ckpt exp/baseline/best.pt` が accuracy と confusion matrix を出力
5. wandb に run が記録される（または offline で run ディレクトリ生成）
6. ベースライン本学習でそれらしい test accuracy（35 クラスで概ね 0.85+ 目安）
7. README の手順だけで第三者が (1)〜(5) を再現できる

## 実装の進め方
- まず ans ブランチで完成版一式（pyproject / README / `src/kws` 完成版 / configs / .gitignore）を作り → スモークで疎通
- main ブランチには `src/kws` と notebook を **TODO 穴埋め版**にして置く（ans の完成版から肝を `TODO` に落とす）
- notebook を各回ぶん執筆（説明厚め＋要所を穴埋め，各回に達成目標，参考ソースの流れを踏襲）

> ファイル名の自動生成みたいな話も盛り込みたい
> パラメータを変えたりするとどう変わるのかまでわかるといいな，最初にパフォーマンスチューニング大会みたいなのをやって，そこから深掘りするのでもいいかも．説明の流れをもっと考えたい．．