# PyTorch 音声コマンド分類 チュートリアル

torchaudio の SPEECHCOMMANDS（35クラス）を題材に，PyTorch で学習パイプラインを一通り組むハンズオン教材．
事前課題 + 全4回．notebook の TODO を埋めながら進め，各回の最後に `src/kws/` へ整理する．

## 資料一覧

| 回 | 資料 | テーマ | 完成の確認 |
|---|---|---|---|
| 事前課題 | [`materials/00_prep.md`](materials/00_prep.md) | 環境構築 + PyTorch 基礎の予習 | `scripts/smoke.py` が PASS |
| 第1回 | [`materials/01_data.ipynb`](materials/01_data.ipynb) | データの読み込み・前処理 | `check_data` PASS |
| 第2回 | [`materials/02_model.ipynb`](materials/02_model.ipynb) | モデル定義（nn.Module） | `check_model` PASS |
| 第3回 | [`materials/03_train.ipynb`](materials/03_train.ipynb) | 学習ループ | `check_train` PASS |
| 第4回 | [`materials/04_eval_tune.ipynb`](materials/04_eval_tune.ipynb) | 学習の深掘り + 評価 + チューニング大会 | `check_eval` PASS |

## 事前準備

[`materials/00_prep.md`](materials/00_prep.md) に従って環境を構築する．

## 進め方

1. `materials/` の notebook を上から実行し，**TODO を埋めて**動かす
2. 各回の最後に，書いたコードを `src/kws/` の対応ファイルに整理する
3. 確認スクリプトを PASS させれば完成

詰まったらメンバー同士の相談 + 主催者の時間外サポートで解決する．

## ディレクトリ構成

```
materials/    資料（prep + 各回の notebook）
src/kws/      学習・評価のコード（TODO 穴埋めスケルトン）
configs/      ハイパーパラメータ（YAML）
scripts/      確認スクリプト
exp/          実験成果物（git 管理外）
data/         SPEECHCOMMANDS（各自 DL，git 管理外）
```

## 参考リンク

- PyTorch "Learn the Basics": https://docs.pytorch.org/tutorials/beginner/basics/intro.html
- torchaudio SPEECHCOMMANDS: https://docs.pytorch.org/audio/stable/generated/torchaudio.datasets.SPEECHCOMMANDS.html
- uv: https://docs.astral.sh/uv/

## 輪講資料
輪講資料はWikiにあります！

[Wikiのページ](https://github.com/shiotalab-tmu/B4_rinko_torchaudio/wiki)