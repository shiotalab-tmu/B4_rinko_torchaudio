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

### 各回の標準フロー

指導者（M）は各回に参加し，来週分の資料を確認し，予習の指示（読む資料・キーワードの提示）を出し，当日の実装を補助する．各回はその上で次の流れで進める．

1. **前回のまとめ**（指導者/サブ指導者が5分程度）．受講者にはやらせない．
2. **予習発表**（受講者）：分担で作った**1スライドを，代表が10〜15分で1本だけ**発表する．一人ずつの個別発表はしない．
   「発表」は前回のまとめではなく，**当回内容の予習**（事前に各自で調べてきたものを持ち寄って1枚にまとめたもの）．
3. **ハンズオン**：前半は md/notebook で説明 → 後半は `src/kws/` の py に実装．各モジュールは**確認スクリプトで PASS** を取って完成を担保する（後述）．
4. **宿題**（2本立て）：
   - **① 実装の続き**：当日終わらなかった人は続きを実装し，「実装の続き ＋ 確認スクリプトの実行結果（PASS） ＋ 学んだこと」を `src/kws/` の対応ファイルに整理してまとめる．
     **輪講の時間内に実装を完結させるのが理想**で，早く終わった人と M が遅れている人の補助に入る．
   - **② 次回の予習**：指導者が指定するサイト・観点を**分担して調べ，1スライドにまとめて次回発表**する．
     発表や院試で忙しい人は予習を軽くしてよいと伝える（予習の重み調整がうまくできていなければ指導者がアドバイスする）．

### 全体マップ

| 回 | テーマ | 主に触るファイル | 完成の確認（これが PASS すれば OK） |
|---|---|---|---|
| 事前課題 | 環境構築 ＋ 資料の予習 | —（環境構築） | スモークスクリプトが通る（import・CUDA・データDL）／予習スライドの分担 |
| 第1回 | キックオフ ＋ PyTorch 基礎 ＋ data | `data.py`（新規） | `check_data`：1バッチが `(B,1,n_mels,T')`・label が 0–34 の `LongTensor` |
| 第2回 | model（`nn.Module`） | `model.py`（新規） | `check_model`：出力 `(B,35)`・初期 loss≈3.56・1バッチ過学習で loss→~0 |
| 第3回 | 学習ループ（最小構成） | `train.py`（新規） | `check_train`：loss が下がる・val acc がチャンス超え・成果物が出力される |
| 第4回 | 学習の深掘り ＋ 評価 ＋ チューニング大会 | `train.py`（改良）＋ `evaluate.py`（新規・小） | `check_eval`：confusion matrix が 35×35・test acc が整合 ／ leaderboard 申告 |

※ 第5回（コンペ独立回）は時間都合で廃止．チューニング大会と leaderboard は**第4回後半で完結**させる．
※ 各 `sessionN_*.md` には，その回で参照する公式チュートリアル等の**関連ドキュメントのリンク**を貼る．

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

※ これらの接続（データの流れ・つまずき点）は，各 `sessionN_*.md` の資料で**図を用いて説明する**．

## 全体を貫く方針

- **確認スクリプトで「作れたか」を機械的に担保する**：各モジュールに，満たすべき仕様を `assert` で書いた確認スクリプト（`check_data` / `check_model` / `check_train` / `check_eval`）を**配る**．
  受講者は実装本体を埋めて，このスクリプトを**全部 PASS させる**ことを目標にする．確認スクリプトは「満たすべき条件（仕様）」を渡すだけで実装本体（答え）は含まないので，**解答非公開と両立**する．
  これは "A Recipe for Training Neural Networks"（Karpathy）の sanity check を教材化したもの：**初期 loss が `ln(35)≈3.56` か**／**1バッチを過学習させて loss→0 になるか**／**データを必ず目で見る**，等を assert にする．
- **解答は非公開**：穴埋め notebook は配るが模範解答（`ans`）は案内しない．確認スクリプトの PASS と，予習発表でのその場の説明で理解を担保する．
- **弱ベースライン**：意図的に単純なモデルにして **val 60〜70% / train 70〜80%** 程度に留め，改善余地を大きく残す．
  underfit 寄りなので「容量を増やす・学習を増やす」施策が素直に効き，因果が分かりやすい．第2回で実装するのもこの弱ベースライン．
  具体 config は実装フェーズで試走確認．
- **wandb は第3回で名前だけ軽く触れる（初出）・必須にしない**．基本の監視手段は `exp/<run>/loss_curve.png` ＋ `history.json`．
  第3回の「進捗を csv に吐く → plot で監視」の流れの中で「tensorboard や wandb というツールもある」と紹介する程度にとどめる．
  アカウント作成・login は強制しない．事前課題では wandb を扱わない（第1回より前にアカウントを作らせない）．
- **test 評価の置き場所**：第3回は val まで ＋ 最後に test accuracy だけ（confusion なし・最小構成）．
  confusion matrix / per-class は**第4回の「分析」で初出**．leaderboard 申告も第4回後半．

## 各回の共通テンプレ

各 `sessionN_*.md` は次の共通テンプレで書く：
**ねらい ／ 関連リンク（その回で参照する公式チュートリアル等） ／ 前回まとめ（指導者が5分） ／
予習発表（受講者が分担で当回内容を予習・1スライド） ／ 座学 ／
ハンズオン（前半 md/notebook で説明 → 後半 py に実装，セル単位・穴埋め粒度） ／
確認スクリプト（`check_*` の assert と期待出力＝PASS の見え方） ／
宿題（実装の続き＋次回予習） ／ 次回への引き継ぎ ／ 速い人向けおまけ**．

## 各回のトップレベル内容

### 事前課題 → `session0_prep.md`
- **環境構築**：**`uv add` から**依存（torch / torchaudio など）を自分で足す ／ SPEECHCOMMANDS のダウンロード（約 2.3GB）．wandb はここでは扱わない．
- **スモークスクリプトで環境を確認**：`import torch, torchaudio` が通り，`torch.__version__`・`torch.cuda.is_available()` を print，
  ついでに `torchaudio.datasets.SPEECHCOMMANDS(root=..., download=True)` を1行叩いてデータ取得まで確認する．
  → これらは**公式 API を呼ぶだけで「答え」を含まない**ので，配布しても解答非公開と矛盾しない．「環境が動く＝import OK・CUDA 見える・データ DL 済み」を一発で確認．
- **全体像を図で先に掴む**：これから4回かけて作る **PyTorch 学習パイプラインの流れ図**（`Dataset` → `DataLoader`（`collate_fn`）→ `Model` → `Loss` → `backward` → `optimizer.step`，評価は `no_grad` で通すだけ）を1枚で見せ，全4回の地図にする．
  図には shape の変化（`waveform` → `(B,1,n_mels,T')` → `(B,35)` → スカラー）と「どの箱を何回目に作るか」（Dataset/DataLoader＝第1回・Model＝第2回・Loss/backward＝第3回・評価＝第4回）を添える．予習範囲（Tensors/DataLoaders）は図の左側に対応すると示す．
  完成形の出力例（confusion matrix・log-mel imshow・loss 曲線の画像）は**第1回当日に見せる**（事前課題では流れ図まで）．ベースラインを各自で動かすには完成版コードが要り，それは受講者が第1〜4回で埋める当のもの＝**配ると解答配布になる**ので動かさない．
- **PyTorch 基礎の予習**：PyTorch 公式 "Learn the Basics" の **Tensors** ／ **Datasets & DataLoaders** の章を読む（第1回 data の予習を兼ねる）．
- 宿題：分担して第1回（data）の予習を1スライドにまとめる ＋ 各自「分かったこと 1 点 ＋ 疑問 1 点」．

### 第1回 — キックオフ ＋ PyTorch 基礎 ＋ data → `session1_kickoff_data.md`
- **予習発表**：事前課題（Tensors / Datasets&DataLoaders）の予習を分担でまとめた1スライドを，10〜15分で1本発表．
- **完成形の出力例を見せる**：弱ベースラインの出力画像（confusion matrix・log-mel imshow・loss 曲線）を貼って，事前課題の流れ図と対応づけて「これを4回かけて作る」と共有する．学習済みモデルをその場で動かす実演（完成形デモ）はしない（解答配布になるため）．
- **PyTorch 基礎**：`torch.tensor` ／ `shape`・`dtype`・`device`．
  適当な Tensor を `.to('cuda')` して `nvidia-smi` で **VRAM 使用量が増えるのを実演**（device の意味を体感）．実際にモデルを載せて使うのは第2回．
  - device やデータの流れは文章だけだと掴みにくいので，**概念図を載せる**（生成画像＝nanobanana 等でも可）．
- **import の仕組み**を伝える：コードを**ファイルに分散して書き，`import` で集約**すると見通しが良くなる（宿題で `data.py` に切り出す動機づけになる）．
- **data**：1サンプル構造 `(waveform, sr, label, speaker_id, utterance_number)`，train/val/test 件数と 35 クラス分布を確認，
  波形 plot・log-mel `imshow`，`label↔index` マップ．
- **バッチサイズの体験**：batch_size を変えると出力 shape の先頭やメモリがどう変わるかを各自いじって観察（モデル幅↔パラメータ数の話は第2回に回す）．
- **Augmentation に言及だけ**入れる（音声の水増しという発想があること．実装は後の布石・速い人向け）．
- **collate_fn**：`pad_or_trim`(16000) → stack `(B,T)` → log-mel → `(B,1,n_mels,T')`．
- **確認スクリプト `check_data`**：DataLoader から 1 バッチ取り出し，`feats.shape == (B,1,n_mels,T')`・`feats.dtype == float32`・`labels` が 0–34 の `LongTensor`・log-mel に NaN/Inf が無い，を assert．全部通れば data 完成．
- 宿題：埋めたコードを `data.py` に整理 ＋ 第2回（model）の予習（分担で1スライド）．
- ※第1回は項目数が多い．**当日のタイムボックスは session1 詳細で要検証**（収まらなければ可視化や分布確認を宿題側へ送る）．

### 第2回 — model → `session2_model.md`
- 前回まとめ：data（指導者/サブ指導者が5分程度）．
- **予習発表：model**（受講者が分担で作った1スライドを10〜15分で1本発表）．
- 座学：`nn.Module`（`__init__` で層を定義／`forward` で繋ぐ），`ConvBlock`（Conv2d-BN-ReLU-MaxPool），GAP→Linear．
  ここで実装するのは**弱ベースライン**（単純なモデル．高性能化は各自のチューニングに残す）．
- **`model.to(device)` の橋渡し**：第1回で触れた device を，ここでモデルに適用する（入力とモデルを同じ device に）．
- shape 追跡（`(B,1,n_mels,T')`→…→`(B,35)`）・**`base` 等モデル幅を変えるとパラメータ数がどう変わるか**を観察．
- **確認スクリプト `check_model`**：ランダム入力 `(B,1,n_mels,T')` → 出力が `(B,35)`／**初期 loss が `ln(35)≈3.56` 付近**／**1バッチを数百 step 回すと loss→ほぼ0**（過学習 sanity check＝配線が正しい）／backward 後にパラメータの `grad` が None でない，を assert．
- 宿題：`model.py` に整理．
- 速い人向け：チューニング向けの独自モジュール追加 ／ もしくは（data 回の復習も兼ねて）Augmentation をどう足すか考える ／ わざと device 不一致エラーを起こしてメッセージを読む．

### 第3回 — 学習ループ（最小構成） → `session3_train.md`
- 前回まとめ：model（指導者が5分）．**予習発表：train（学習ループ）**（受講者が分担で予習・1スライド）．
- 座学：`CrossEntropyLoss`（生 logits を渡す）／`Adam`／学習の5行 `zero_grad → forward → loss → backward → step`．
- **ストーリー仕立てで組み立てる**：まず素朴に学習ループを回す → 「今が何 epoch 目か・うまくいっているか分からない」と気づく →
  進捗を csv（`history.json`）に吐く → 吐いておけば後から plot して監視できる（`loss_curve.png`）→
  **ここで「tensorboard や wandb というツールもある」と軽く触れる**（初出・紹介程度・必須にしない）．
- `run_epoch`（train/val 共通），固定エポック，accuracy．
- **評価の作法**：val/test は `model.eval()` ＋ `torch.no_grad()` で回す（勾配を流さない・BN統計を更新しない）．
- 学習後に `train.py` 内で test loader を1回まわして **test accuracy だけ**出す（`evaluate.py` はまだ作らない・confusion も持ち込まない）．
- **確認スクリプト `check_train`**：数 epoch 回して train loss が下がる／val acc がチャンスレート（`1/35≈2.9%`）を超える／`history.json` と `loss_curve.png` が生成される／最後に test accuracy が弱ベースライン水準で出る，を assert．
- 宿題：`train.py` に整理．

### 第4回 — 学習の深掘り ＋ 評価 ＋ チューニング大会 → `session4_deepdive_eval.md`
- 前回まとめ：train（指導者が5分）．**予習発表：評価/チューニング**（overfit/underfit・scheduler・評価指標など，受講者が分担で予習・1スライド）．
- **前半（深掘り・手を動かす）**：第3回の loss 曲線を起点に overfit/underfit の読み方，early stopping，lr scheduler，learning rate，seed・再現性．
- **評価（見せる／分析）**：`evaluate.py` を**新規作成するが穴は小さく**（受講者が書くのは「全バッチの pred/label を集める」predict だけ．
  confusion matrix の描画と classification report は用意済みを呼ぶだけ）．これで「どのクラスを間違えるか」を分析する．
- **確認スクリプト `check_eval`**：predict の出力長が test 件数と一致／test acc が学習時に見た値と整合／confusion matrix が 35×35，を assert．
- **後半：チューニング大会**．道具（scheduler・lr・容量・Aug 等）が揃ったので，各自その場で工夫して弱ベースライン超えを狙う． 
  各自 `evaluate.py` で test を評価 → **簡易 leaderboard（Slack 等）に申告**して締める．
  test split は torchaudio 公式で全員共通固定，指標＝accuracy．test リークは教育目的で許容（「やってはいけないこと」も含めて学ぶ）．

## 作成の進め方

トップレベル（本ファイル）→ 事前課題 → 第1回 → … と **1単位ずつ「調査 → ドラフト → 多角検証 → 提示」** で固める．
各単位は確定したら詳細 md を作成し，index と矛盾がないかを毎回突き合わせる．
