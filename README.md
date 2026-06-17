# PyTorch 音声情報処理チュートリアル（SPEECHCOMMANDS / KWS）

PyTorch で深層学習による音声情報処理を一通り体験するハンズオン教材．
**環境構築（uv）→ データローダー → モデル → 学習ループ → 評価・チューニング**までを，
torchaudio の `SPEECHCOMMANDS`（35 クラスのキーワードスポッティング, KWS）を題材に学ぶ．

事前課題 ＋ 全 4 回．notebook を教科書として進め，各回の最後に `src/kws/` の python ファイルへ整理する．
最終的に「python ファイル・実験結果・notebook」が手元に残り，実験が再現できる状態を目指す．

各回の詳細は `docs/sessions/`（地図＝`index.md`，各回詳細＝`sessionN_*.md`）を参照．

## 進め方（重要）
- **notebook が本体**．`notebooks/0X_*.ipynb` を上から実行し，説明を読みつつ **TODO（穴埋め）** を埋めて動かす．
- 各回の最後に，穴埋めしたコードを **`src/kws/` の対応ファイルに整理**する（こちらも TODO スケルトン）．
- 各モジュールには**確認スクリプト `check_*`**（満たすべき仕様を `assert` で書いたもの）が付く．これを **全部 PASS** させれば完成．
- **模範解答は配布・案内しない**．詰まったらメンバー同士の相談 ＋ 主催者の時間外サポートで解決する．

## 事前準備（事前課題）
詳しい手順は `docs/sessions/session0_prep.md`．要点だけ：

1. **uv のインストール**（パッケージ管理）: https://docs.astral.sh/uv/getting-started/installation/
2. **uv プロジェクトをゼロから作る**（このリポジトリには `pyproject.toml` を含めない）:
   ```bash
   uv init --python 3.12
   uv add "torch==2.4.1" "torchaudio==2.4.1" matplotlib
   ```
   - torch の PyPI wheel（Linux）は CUDA ランタイム同梱なので，素の `uv add` で CUDA 版が入る．
3. **データセットの取得**（SPEECHCOMMANDS v0.02, 約 2.3GB）:
   ```bash
   uv run python -c "import torchaudio; torchaudio.datasets.SPEECHCOMMANDS(root='data', download=True)"
   ```
4. **環境チェック**（スモーク）:
   ```bash
   uv run python scripts/smoke.py
   ```
   - wandb は事前課題では使わない（第3回で名前だけ軽く触れる程度）．

## 実行方法（CLI）
notebook で流れを掴んだあとは，整理した python ファイルを CLI で回す．

```bash
# 学習（exp/<run_name>/ に last.pt=現在の重み / best.pt=val最良 を保存）
uv run python -m kws.train --config configs/baseline.yaml --device cuda

# 中断したら last.pt から再開
uv run python -m kws.train --config configs/baseline.yaml --device cuda --resume

# スモーク（1 epoch だけ・wandb なし）※ baseline を汚さないよう別 run-name にする
uv run python -m kws.train --config configs/baseline.yaml --device cuda --epochs 1 --no-wandb --run-name smoke

# 評価（accuracy / confusion matrix を出力・保存）
uv run python -m kws.evaluate --ckpt exp/baseline/best.pt --device cuda

# 学習→評価を一気に
bash scripts/run_baseline.sh
```

## 各回の概要
| 回 | テーマ | 対象ファイル | 完成の確認 |
|---|---|---|---|
| 事前課題 | 環境構築 ＋ 予習 | —（環境構築） | スモークが PASS する |
| 1 | PyTorch 基礎 + データ | `src/kws/data.py` | `check_data`：DataLoader から `(B,1,n_mels,T')` のバッチが出る |
| 2 | モデル（nn.Module） | `src/kws/model.py` | `check_model`：ランダム入力→`(B,35)` logits / 1バッチ過学習で loss→~0 |
| 3 | 学習ループ（最小構成） | `src/kws/train.py` | `check_train`：loss が下がる / test acc が出る |
| 4 | 学習の深掘り + 評価 + チューニング大会 | `src/kws/train.py`・`evaluate.py` | `check_eval`：test acc + 混同行列 / leaderboard 申告 |

## ディレクトリ構成
```
src/kws/      学習・評価のコード（TODO 穴埋めスケルトン）
configs/      ハイパーパラメータ（YAML）
notebooks/    教科書（TODO 穴埋め式）
scripts/      確認スクリプト・再現用スクリプト
exp/          実験成果物（checkpoint, figures, metrics; 中身は git 管理外）
data/         SPEECHCOMMANDS（各自 DL; git 管理外）
docs/         勉強会の設計・各回の詳細（sessions/）
```

## 参考ソース
- PyTorch 公式 "Learn the Basics": https://docs.pytorch.org/tutorials/beginner/basics/intro.html
- torchaudio "Speech Command Classification"（Colab）: https://colab.research.google.com/github/pytorch/tutorials/blob/gh-pages/_downloads/c64f4bad00653411821adcb75aea9015/speech_command_classification_with_torchaudio_tutorial.ipynb
- wandb PyTorch integration: https://docs.wandb.ai/models/integrations/pytorch
- uv: https://docs.astral.sh/uv/
