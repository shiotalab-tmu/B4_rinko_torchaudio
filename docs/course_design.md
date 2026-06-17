# 今回決めたこと（講義設計）

> このドキュメントは、議論を経て **確定した全 4 回の構成と各回の引き継ぎ・大まかな内容・方針**をまとめたもの。
> 各回の具体的な内容（地図）は `docs/sessions/index.md` を正とする。
> - 運営の進め方（1 回のテンプレート・発表の運用・巡回・Tips の散らし方）は `docs/operation.md`。
> - 次に何をすればよいか（各回の詳細をどう詰めるか）は `docs/handoff.md`。
>
> 各回で当日「具体的に何を話す・何を実装させるか」の細部はまだ詰めている途中。確定したものから順にここへ書き足す。

## 大目標と受講者像（確定）

- 大目標 = **PyTorch で動く「学習・評価スクリプト」を自分で書けるようになること**。
  音声コマンド認識（torchaudio SPEECHCOMMANDS, 35 クラス, log-mel + 小型 2D-CNN）は **題材＝手段であって目標ではない**。
- 受講者 = 研究室 B4 が 5〜6 名。「ゼロから作る Deep Learning」で理論は履修済みだが、自分でコードを書いた経験が乏しい。
  → 理論の再説明・内部の深掘りはしない。「PyTorch でどう書くか」と「全体の流れ」に集中する。

## 全体構成（確定）

全 4 回 ＋ 事前課題。ボトムアップで「動くパイプライン」を 1 モジュールずつ積む（data → model → 学習 → 評価・チューニング）。
毎回 **「前回まとめ（指導者が 5 分）＋ 当回内容の予習発表（受講者）→ 当日ハンズオン → 宿題で `src/kws/` に整理」** のイテレーションで進む。
第 1 回も予習発表ありにして全回を同じ型にした。学習パートは過密回避のため **実装回（第 3 回）と Tips・深掘り回（第 4 回）に分離**。

「発表」は前回のまとめではなく **当回内容の予習**（各自が事前に調べたものを分担で 1 スライドにまとめ、代表が 10〜15 分で 1 本だけ発表する）。

| 回 | 予習発表（当回分） | 当日やる | 宿題（次回への引き継ぎ） |
|---|---|---|---|
| 事前課題 | — | （各自）環境構築 ＋ 資料の予習 | 第 1 回（data）の予習を分担で 1 スライド ＋「分かったこと 1 点 ＋ 疑問 1 点」 |
| 第 1 回 | Tensors / DataLoaders | PyTorch 基礎 ＋ data | data.py に整理 ＋ 第 2 回の予習 |
| 第 2 回 | model | model | model.py に整理 ＋ 第 3 回の予習 |
| 第 3 回 | 学習ループ | 学習・評価の実装（固定エポックで回して最後に評価するだけの最小構成） | train.py に整理 ＋ 第 4 回の予習 |
| 第 4 回 | 評価・チューニング | 学習の深掘り（early stopping / 監視 / scheduler / lr / seed・再現性）＋ 評価（`evaluate.py`）＋ チューニング大会・leaderboard | チューニング |

※ 第 5 回（コンペ独立回）は時間都合で廃止し、チューニング大会と leaderboard は **第 4 回後半で完結**させる。

設計のポイント：

- **第 3 回（実装回）** は「学習ループ（zero_grad → forward → loss → backward → step）＋ 固定エポック ＋ 最後に accuracy 評価」の最小構成に絞る。best/last 切替・early stopping・scheduler・confusion matrix は持ち込まない（過密回避）。
- **第 4 回（深掘り回）** で、第 3 回で見た loss 曲線を起点に early stopping / 監視 / scheduler / lr / seed・再現性を入れ、`evaluate.py` で confusion matrix・per-class を分析し、そのまま「これらを使って各自チューニング」へ号砲してその場で leaderboard 申告まで完結する。
- 着地点（完成形の出力例）は **事前課題で各自が画像として見る**（confusion matrix の png・log-mel の imshow・loss 曲線）。第 1 回当日の完成形デモは行わない。

## 各回の大まかな内容

各回の概要を示す。**当日の詳しい流れ・穴埋めの粒度・確認スクリプトの中身は `docs/sessions/` で詰める。** 第 1 回のみ詳しめ。

### 第 1 回 — キックオフ（PyTorch 基礎 ＋ データ） / 対象: `src/kws/data.py`

1. **予習発表**: 事前課題（Tensors / Datasets&DataLoaders）の予習を分担でまとめた 1 スライドを代表が発表
2. **PyTorch 基礎ハンズオン**: `torch.tensor` / `shape`・`dtype`・`device` / `x.to(device)` と「同じ device に揃える」
3. **データを触る**: 1 サンプル構造 `(waveform, sr, label)`、データ内訳（train / val / test の件数・35 クラス分布）を確認して資産化、波形と log-mel を imshow、label↔index マップ
4. **`collate_fn` で 1 秒（16000）固定 → DataLoader で `(B,1,n_mels,T)` を出す** ←【確認スクリプト `check_data`】

- 当日は **log-mel まで全部見せる**。宿題は埋めたコードを `data.py` に整理すること。
- 環境構築は事前課題に含める（当日は動作確認のみ）。
- 達成目標: DataLoader を回すと `(B,1,n_mels,T)` のミニバッチが出てくることを確認できる。

### 第 2 回 — モデル（`model.py`）

- `nn.Module` の書き方（`__init__` で層を定義 / `forward` で繋ぐ）、`AudioCNN`（log-mel 入力の 2D-CNN）の forward。
- 入出力 shape の追跡・パラメータ数。1 バッチ過学習の sanity check で「配線が正しい」ことを確認。
- 達成目標: ランダム入力 → 35 次元 logits ／ 1 バッチ過学習で loss → ほぼ 0。（詳細は次フェーズ）

### 第 3 回 — 学習・評価の実装（`train.py`、最小構成）

- 学習ループ: `optimizer.zero_grad()` → forward → loss → `loss.backward()` → `optimizer.step()`、train/val、固定エポック。
- 最後に accuracy で評価するだけ。**best/last 切替・early stopping・scheduler・confusion matrix は持ち込まない**（次回に回す）。
- loss 曲線を見る（第 4 回の深掘りの入口）。
- 達成目標: 学習が回って loss が下がる ／ test accuracy が出る。（詳細は次フェーズ）

### 第 4 回 — 学習の深掘り ＋ 評価 ＋ チューニング大会

- **前半（深掘り）**: 第 3 回で見た loss 曲線を起点に **early stopping / 実験の監視 / lr scheduler / learning rate / seed・再現性**。
- **評価**: `evaluate.py` を新規作成（穴は小さく＝受講者が書くのは pred/label を集める predict だけ。confusion matrix 描画と classification report は用意済みを呼ぶ）。「どのクラスを間違えるか」を分析する（confusion matrix・per-class はここで初出）。
- **後半（チューニング大会）**: 道具が揃ったので各自その場で弱ベースライン超えを狙い、各自 `evaluate.py` で test 評価 → 簡易 leaderboard に申告して締める。test split は torchaudio 公式で全員共通固定、指標＝accuracy。
- 宿題 ＝ チューニング。（詳細は `docs/sessions/session4_deepdive_eval.md`）

## 解答の扱い（丸写し対策・確定）

- **解答は公開しない**。`ans` ブランチに置くが「クローンすれば答え」という案内はしない。丸写しする層はそもそも別ブランチを取りに来られないので実質防げる。
- 詰まったら **研究室メンバー同士で相談** ＋ 主催者が時間外にも相談に乗る。
- 「動く」を曖昧にしないため、各モジュールに **確認スクリプト `check_*`**（満たすべき仕様を `assert` で書いたもの）を配り、受講者は実装本体を埋めて **全部 PASS させる**ことを目標にする。スクリプトは仕様だけで実装本体（答え）を含まないので、解答非公開と両立する。Karpathy "A Recipe for Training Neural Networks" の sanity check（初期 loss が `ln(35)≈3.56` か／1 バッチ過学習で loss→0 か／データを目で見る）を教材化したもの。

## コンペ・評価・baseline（確定）

- **弱 baseline**: 意図的に単純なモデルにし、**val 60〜70% / train 70〜80%** 程度に留める（改善余地を大きく残す）。
  underfit 型なので「容量を増やす・学習を増やす」施策が素直に効き、因果が分かりやすい。配布前に手元で 1 回試走して数値を確認する。
- **test リーク**: 教育目的なので許容（チューニング中に test を使ってしまうのも「やってはいけないことの学び」）。
- **テスト評価**: 各自が `evaluate.py` で test を回して申告。test split は torchaudio 公式で全員共通固定。最終的な統一評価をどうするかは終盤に検討。

## 確定した個別の決定事項

- 第 1 回も予習発表ありにする（事前に資料を読み、当回内容の予習を分担で 1 スライドにまとめて発表。各自「分かったこと ＋ 疑問」も持ち寄る）。
- 環境構築（`uv add` で依存を自分で足す・データ DL）は **事前課題（初回の宿題）** に含める。当日は動作確認のみ。**wandb は事前課題では扱わない**（第 3 回で名前だけ軽く触れる・login は強制しない）。
- 第 1 回の data は **当日に log-mel まで全部見せる**。宿題は `data.py` への整理。
- 学習パートを実装回（第 3 回）と深掘り回（第 4 回）に分離。
- 宿題の難易度はメンバー同士の相談＋主催者の時間外サポートで吸収する（部分解の配布などはしない）。
