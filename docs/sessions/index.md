# 各回の具体的内容 — トップレベル（箇条書き）

> このファイルは **各回に盛り込む具体的内容を箇条書きでマネジメントする作業ボード**。
> ここで全体像（各回に何をどんな順で入れるか）を対話的に固め、固まった回から
> `docs/sessions/sessionN_*.md` に超詳細（読めば誰でもノートブックを作れるレベル）を書く。
>
> - 上位の確定情報（テーマ・全体パターン）: `../course_design.md`
> - 運営の流れ: `../operation.md`
> - 経緯と展望: `../handoff.md`
>
> 記法: `[確定]` = 議論で決まった / `[要詰め]` = これから対話で決める / `[案]` = 叩き台

## 第 1 回 — data（→ `session1_data.md`）

- `[確定]` 環境構築は事前課題。当日は動作確認のみ。
- `[確定]` data は当日 log-mel まで全部見せる。宿題は `data.py` への整理。
- `[案]` 当日の流れ: 完成形デモ → 事前学習の発表 → PyTorch 基礎（tensor / shape・dtype・device / `.to(device)`）
  → データを触る（1 サンプル構造・データ内訳・波形と log-mel の可視化・label↔index）→ `collate_fn` で 1 秒固定 → DataLoader で `(B,1,n_mels,T)`
- `[要詰め]` 事前学習で読ませる資料の指定（範囲・最低ライン）
- `[要詰め]` `data.py` の穴埋め粒度（どこを見せて、どこを書かせるか）
- `[要詰め]` 動作確認例題の中身と期待出力

## 第 2 回 — model（→ `session2_model.md`）

- `[案]` `nn.Module` の書き方 → `AudioCNN` の forward → shape 追跡・パラメータ数 → 1 バッチ過学習の sanity check
- `[要詰め]` 当日の流れ・穴埋め粒度・動作確認例題

## 第 3 回 — 学習・評価の実装（最小構成）（→ `session3_train.md`）

- `[確定]` 固定エポックで回して最後に accuracy 評価するだけ。best/last 切替・early stopping・scheduler・confusion は持ち込まない。
- `[案]` 学習ループ（zero_grad → forward → loss → backward → step）→ train/val → loss 曲線を見る → test accuracy
- `[要詰め]` 当日の流れ・穴埋め粒度・動作確認例題

## 第 4 回 — 学習の重要事項の深掘り ＋ チューニング号砲（→ `session4_tips.md`）

- `[案]` 第 3 回の loss 曲線を起点に: early stopping / 実験の監視 / lr scheduler / learning rate / seed・再現性 → チューニングへ
- `[要詰め]` confusion matrix / per-class をここで分析として扱うか
- `[要詰め]` 深掘り項目の見せ方・順序・動作確認例題

## 第 5 回 — チューニング結果発表 ＋ コンペ（→ `session5_competition.md`）

- `[案]` 各自の改善結果を発表 → 各自 `evaluate.py` で test 評価して申告 → 簡易 leaderboard
- `[要詰め]` leaderboard の運用（スプレッドシート等）・評価指標
- `[要詰め]` 弱 baseline の具体 config（val 60〜70% / train 70〜80%。試走で確認）
