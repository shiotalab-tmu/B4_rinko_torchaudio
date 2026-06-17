# 各回の内容（トップレベル一覧）

このフォルダ `docs/sessions/` は，**各回で当日具体的に何を話し，何を実装させるか**を，
「読めば誰でも同じノートブックを起こせる」レベルで書き下すためのもの．
本ファイル（`index.md`）はその**地図**で，全回を共通の型で一覧化し，過密・重複・受け渡しの破綻を防ぐ役割を持つ．
各回の超詳細は `sessionN_*.md` に分ける．

- 上位の確定事項（大目標・全体構成・方針）は `docs/course_design.md`．
- 運営の進め方（1回のテンプレート・発表・巡回）は `docs/operation.md`．
- ここまでの経緯は `docs/handoff.md`．

## 全体像

- **大目標**：受講者が「PyTorch で動く学習・評価スクリプトを自分で書けるようになる」こと．
  音声コマンド認識（torchaudio SPEECHCOMMANDS, 35クラスの KWS, log-mel 特徴 ＋ 小型 2D-CNN）は**題材＝手段**．
- **受講者**：研究室 B4 が 5〜6 名．「ゼロから作る Deep Learning」で理論は履修済みだが，自分でコードを書いた経験が乏しい．
  → 理論の再説明・内部の深掘りはしない．「PyTorch でどう書くか」と「全体の流れ」に集中する．
- **構成**：事前課題 ＋ **全4回**．ボトムアップに `data → model → 学習 → 評価/チューニング` を 1 モジュールずつ積む．
  各回は「前回モジュールの発表 → 当日ハンズオン（前半 md/notebook で説明 → 後半 py に実装） → 宿題で `src/kws/` に整理」のイテレーション．

| 回 | テーマ | 主に触るファイル | 動作確認例題（これが動けば OK） |
|---|---|---|---|
| 事前課題 | 環境構築 ＋ ベースライン推論 ＋ 資料読み | —（環境構築） | 配布ベースラインで推論 → 結果画像が出る ／「分かったこと1＋疑問1」 |
| 第1回 | キックオフ ＋ PyTorch 基礎 ＋ data | `data.py`（新規） | DataLoader から `(B,1,n_mels,T')` のバッチが出る |
| 第2回 | model（`nn.Module`） | `model.py`（新規） | ランダム入力→`(B,35)` logits ／ 1バッチ過学習で loss→ほぼ0 |
| 第3回 | 学習ループ（最小構成） | `train.py`（新規） | 学習が回り loss が下がる ／ test accuracy が弱ベースライン水準で出る |
| 第4回 | 学習の深掘り ＋ 評価 ＋ チューニング大会 | `train.py`（改良）＋ `evaluate.py`（新規・小） | `evaluate.py` で test acc ＋ confusion matrix が出る ／ leaderboard 申告 |

※ 第5回（コンペ独立回）は時間都合で廃止．チューニング大会と leaderboard は**第4回後半で完結**させる．

## モジュール間の受け渡し（全体を貫く「型」）

全回はこの 1 本のデータの流れを少しずつ作っている．各回の詳細はこの型と矛盾しないこと．

```
data :  (waveform, sr, label, …)  →  1秒(16000)に揃える  →  stack (B, T)  →  log-mel  →  (B, 1, n_mels, T')
model:  (B, 1, n_mels, T')  →  ConvBlock × N  →  GAP  →  Linear  →  (B, 35) logits
train:  logits (B,35) ＋ label index (B,)  →  CrossEntropyLoss  →  backward  →  optimizer.step()
eval :  全バッチの logits→argmax(pred) と label を蓄積  →  accuracy ／ confusion matrix ／ per-class
```

※ `eval` 行は第3回（accuracy だけ・`train.py` 内）→ 第4回（`evaluate.py` で confusion / per-class を追加）に分けて作る．

接続上，特に資料で**明示すべき点**（初学者がつまずく定番）：

- **ラベルは index の `LongTensor`**（one-hot ではない）．第1回で作る `label↔index` マップが，そのまま第3回の `CrossEntropyLoss` に渡る．
- **`CrossEntropyLoss` には softmax をかけない生の logits を渡す**（内部で log_softmax されるため，二重に softmax すると学習が劣化する）．
- 入力の時間長 `T'` が可変でも **GAP（Global Average Pooling）で 1 点に潰す**ので，モデルは `T'` に依存しない．「だから GAP を使う」と一言添える．

## 全体を貫く方針

- **wandb は第3回で名前だけ軽く触れる（初出）・必須にしない**．基本の監視手段は `exp/<run>/loss_curve.png` ＋ `history.json`．
  第3回の「進捗を csv に吐く → plot で監視」の流れの中で「tensorboard や wandb というツールもある」と紹介する程度にとどめる．
  アカウント作成・login は強制しない．事前課題では wandb を扱わない（第1回より前にアカウントを作らせない）．
- **弱ベースライン**：意図的に単純なモデルにして **val 60〜70% / train 70〜80%** 程度に留め，改善余地を大きく残す．
  underfit 寄りなので「容量を増やす・学習を増やす」施策が素直に効き，因果が分かりやすい．第2回で実装するのもこの弱ベースライン．
  具体 config は実装フェーズで試走確認．
- **解答は非公開**：穴埋め notebook は配るが模範解答（`ans`）は案内しない．代わりに各モジュールに
  **「これが動けば OK」の動作確認例題**（スニペット ＋ 期待出力）を置き，発表でその場で実演させて担保する．
- **test 評価の置き場所**：第3回は val まで ＋ 最後に test accuracy だけ（confusion なし・最小構成）．
  confusion matrix / per-class は**第4回の「分析」で初出**．leaderboard 申告も第4回後半．

## 各回のトップレベル内容

各 `sessionN_*.md` は次の共通テンプレで書く：
**ねらい ／ 事前に読む資料 ／ 発表（前回分） ／ 座学 ／
ハンズオン（前半 md/notebook で説明 → 後半 py に実装，セル単位・穴埋め粒度） ／
動作確認例題（スニペット＋期待出力） ／ 宿題 ／ 次回への引き継ぎ ／ 速い人向けおまけ**．

宿題は全回共通の型：**前半で説明したものを後半で py に実装し，終わらなかった人は続きを宿題に回す．
提出物＝「実装の続き ＋ 動作確認コードの実行結果 ＋ 学んだこと」**を `src/kws/` の対応ファイルに整理してまとめる．

### 事前課題 → `session0_prep.md`
- 環境構築：**`uv add` から**依存（torch / torchaudio など）を自分で足す ／ SPEECHCOMMANDS のダウンロード（約 2.3GB）．wandb はここでは扱わない．
- **着地点を先に見せる**：配布したベースライン（学習済み）で**推論を1回流して結果画像が出れば成功**．
  「最終的にこういうものを自分で作る」というゴールを最初に体感させる．
- 指定資料を読む：PyTorch 公式 "Learn the Basics" の **Tensors** ／ **Datasets & DataLoaders** の章．
- 宿題：「分かったこと 1 点 ＋ 疑問 1 点」を第1回の発表用にまとめる．

### 第1回 — キックオフ ＋ PyTorch 基礎 ＋ data → `session1_kickoff_data.md`
- **事前課題の発表**（各自 2〜3 分）から入る．着地点は各自が事前課題で見ているので，当日の完成形デモは行わない．
- **PyTorch 基礎**：`torch.tensor` ／ `shape`・`dtype`・`device`．
  適当な Tensor を `.to('cuda')` して `nvidia-smi` で **VRAM 使用量が増えるのを実演**（device の意味を体感）．実際にモデルを載せて使うのは第2回．
- **data**：1サンプル構造 `(waveform, sr, label, speaker_id, utterance_number)`，train/val/test 件数と 35 クラス分布を確認，
  波形 plot・log-mel `imshow`，`label↔index` マップ．
- **バッチサイズの体験**：batch_size を変えると出力 shape の先頭やメモリがどう変わるかを各自いじって観察（モデル幅↔パラメータ数の話は第2回に回す）．
- **Augmentation に言及だけ**入れる（音声の水増しという発想があること．実装は後の布石・速い人向け）．
- **collate_fn**：`pad_or_trim`(16000) → stack `(B,T)` → log-mel → `(B,1,n_mels,T')`．
- 当日は **log-mel まで全部見せる**．動作確認例題＝DataLoader から 1 バッチ取り出して `(B,1,n_mels,T')` を確認．
- 宿題：埋めたコードを `data.py` に整理．
- ※第1回は発表回かつ最も項目数が多い．**当日のタイムボックスは session1 詳細で要検証**（収まらなければ可視化や分布確認を宿題側へ送る）．

### 第2回 — model → `session2_model.md`
- 発表：data（**頭30分＝6人×5分**を発表枠にとる）．
- 座学：`nn.Module`（`__init__` で層を定義／`forward` で繋ぐ），`ConvBlock`（Conv2d-BN-ReLU-MaxPool），GAP→Linear．
  ここで実装するのは**弱ベースライン**（単純なモデル．高性能化は各自のチューニングに残す）．
- **`model.to(device)` の橋渡し**：第1回で触れた device を，ここでモデルに適用する（入力とモデルを同じ device に）．
- shape 追跡（`(B,1,n_mels,T')`→…→`(B,35)`）・**`base` 等モデル幅を変えるとパラメータ数がどう変わるか**，**1バッチ過学習の sanity check**（配線が正しいか）．
- 動作確認例題：ランダム入力 `(B,1,64,T')` → `(B,35)` logits ／ 1バッチを数百 step 回して loss→ほぼ0．
- 宿題：`model.py` に整理．
- 速い人向け：チューニング向けの独自モジュール追加 ／ もしくは（data 回の復習も兼ねて）Augmentation をどう足すか考える ／ わざと device 不一致エラーを起こしてメッセージを読む．

### 第3回 — 学習ループ（最小構成） → `session3_train.md`
- 発表：model．
- 座学：`CrossEntropyLoss`（生 logits を渡す）／`Adam`／学習の5行 `zero_grad → forward → loss → backward → step`．
- **ストーリー仕立てで組み立てる**：まず素朴に学習ループを回す → 「今が何 epoch 目か・うまくいっているか分からない」と気づく →
  進捗を csv（`history.json`）に吐く → 吐いておけば後から plot して監視できる（`loss_curve.png`）→
  **ここで「tensorboard や wandb というツールもある」と軽く触れる**（初出・紹介程度・必須にしない）．
- `run_epoch`（train/val 共通），固定エポック，accuracy．
- **評価の作法**：val/test は `model.eval()` ＋ `torch.no_grad()` で回す（勾配を流さない・BN統計を更新しない）．
- 学習後に `train.py` 内で test loader を1回まわして **test accuracy だけ**出す（`evaluate.py` はまだ作らない・confusion も持ち込まない）．
- 動作確認例題：学習が回り loss が下がる ／ test acc が弱ベースライン水準で出る．
- 宿題：`train.py` に整理．

### 第4回 — 学習の深掘り ＋ 評価 ＋ チューニング大会 → `session4_deepdive_eval.md`
- 発表：train．
- **前半（深掘り・手を動かす）**：第3回の loss 曲線を起点に overfit/underfit の読み方，early stopping，lr scheduler，learning rate，seed・再現性．
- **評価（見せる／分析）**：`evaluate.py` を**新規作成するが穴は小さく**（受講者が書くのは「全バッチの pred/label を集める」predict だけ．
  confusion matrix の描画と classification report は用意済みを呼ぶだけ）．これで「どのクラスを間違えるか」を分析する．
- **後半：チューニング大会**．道具（scheduler・lr・容量・Aug 等）が揃ったので，各自その場で工夫して弱ベースライン超えを狙う．
  各自 `evaluate.py` で test を評価 → **簡易 leaderboard（Slack 等）に申告**して締める．
  test split は torchaudio 公式で全員共通固定，指標＝accuracy．test リークは教育目的で許容（「やってはいけないこと」も含めて学ぶ）．
- 動作確認例題：`evaluate.py` で test acc ＋ confusion matrix が出る．

## 作成の進め方

トップレベル（本ファイル）→ 事前課題 → 第1回 → … と **1単位ずつ「調査 → ドラフト → 多角検証 → 提示」** で固める．
各単位は確定したら詳細 md を作成し，index と矛盾がないかを毎回突き合わせる．
