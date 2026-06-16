# PyTorch 音声情報処理チュートリアル（SPEECHCOMMANDS / KWS）

PyTorch で深層学習による音声情報処理を一通り体験するハンズオン教材．
**環境構築（uv）→ データローダー → モデル → 学習ループ → 評価 → wandb** までを，
torchaudio の `SPEECHCOMMANDS`（35 クラスのキーワードスポッティング, KWS）を題材に学ぶ．

全 5 回．notebook を教科書として進め，各回の最後に `src/kws/` の python ファイルへ整理する．
最終的に「python ファイル・実験結果・notebook」が手元に残り，実験が再現できる状態を目指す．

## 進め方（重要）
- **notebook が本体**．`notebooks/0X_*.ipynb` を上から実行し，説明を読みつつ **TODO（穴埋め）** を埋めて動かす．
- 各回の最後に，穴埋めしたコードを **`src/kws/` の対応ファイルに整理**する（こちらも TODO スケルトン）．
- 詰まったら模範解答を `ans` ブランチから取り出せる:
  ```bash
  git checkout ans -- src/kws/data.py   # 例: data.py の完成版だけ取得
  ```
  - `main`：受講者が穴埋めする版（スケルトン）
  - `ans`：完成版（模範解答）を回ごとに 1 コミットずつ収録

## 事前準備（授業の前に各自すませておく）
重いダウンロードを授業時間から外すため，**事前に**以下を済ませておくと安心．

1. **uv のインストール**（パッケージ管理）: https://docs.astral.sh/uv/getting-started/installation/
2. **依存のインストール**（torch/torchaudio など．`pyproject.toml` の通り）:
   ```bash
   uv sync
   ```
   - `pyproject.toml` は CUDA(cu121) ホイールを使う設定．GPU マシンの CUDA に合わせて
     `cu118` / `cu124` などへ変更してよい．CPU だけで試すなら該当 index を外す．
3. **データセットの取得**（SPEECHCOMMANDS v0.02, 約 2.3GB）:
   ```bash
   mkdir -p data
   uv run python -c "import torchaudio; torchaudio.datasets.SPEECHCOMMANDS(root='data', download=True)"
   ```
4. **wandb のアカウント作成と login**（第3回まで）:
   ```bash
   uv run wandb login
   ```
   - アカウントを作らない場合は `WANDB_MODE=offline` で実行するか，学習時に `--no-wandb` を付ける．

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
| 回 | テーマ | 対象ファイル | 達成目標 |
|---|---|---|---|
| 1 | PyTorch 超入門 + データ | `src/kws/data.py` | DataLoader から log-mel バッチが出る |
| 2 | モデル（nn.Module） | `src/kws/model.py` | ランダム入力→logits / 1バッチ過学習 |
| 3 | 学習ループ + wandb | `src/kws/train.py` | loss 曲線が wandb に出る |
| 4 | 評価と再現性 | `src/kws/evaluate.py` | test acc と混同行列が出る |
| 5 | 性能向上ワークショップ | `configs/*.yaml` | 自分の工夫でベースライン超え |

## ディレクトリ構成
```
src/kws/      学習・評価のコード（main=スケルトン / ans=完成版）
configs/      ハイパーパラメータ（YAML）
notebooks/    教科書（TODO 穴埋め式）
scripts/      再現用スクリプト
exp/          実験成果物（checkpoint, figures, metrics; 中身は git 管理外）
data/         SPEECHCOMMANDS（各自 DL; git 管理外）
```

## 参考ソース
- PyTorch 公式 "Learn the Basics": https://docs.pytorch.org/tutorials/beginner/basics/intro.html
- torchaudio "Speech Command Classification"（Colab）: https://colab.research.google.com/github/pytorch/tutorials/blob/gh-pages/_downloads/c64f4bad00653411821adcb75aea9015/speech_command_classification_with_torchaudio_tutorial.ipynb
- wandb PyTorch integration: https://docs.wandb.ai/models/integrations/pytorch
- uv: https://docs.astral.sh/uv/
