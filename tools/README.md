# tools/ — notebook ジェネレータ（保守用）

`notebooks/*.ipynb` は手書きではなく nbformat で生成している．1 つの定義から
**完成版（full / `ans` ブランチ用）** と **穴埋め版（blank / `main` ブランチ用）** を出し分ける．
notebook を直すときは `.ipynb` ではなく **ここの `gen_*.py` を編集して再生成**する方が，
完成版と穴埋め版がずれず安全．

## ファイル
- `nbh.py` … 共通ヘルパ（`md` / `code` / `sol(full, blank)` / `build`）．`sol` が穴埋め対象セル．
- `gen_0X_*.py` … 各回の notebook 定義．
- `regen.sh` … 全 notebook を一括生成．

## 使い方
個別:
```bash
uv run python tools/gen_01_data.py notebooks/01_data.ipynb full   # 完成版
uv run python tools/gen_01_data.py notebooks/01_data.ipynb blank  # 穴埋め版
```
一括:
```bash
bash tools/regen.sh full     # 完成版（ans 用）
bash tools/regen.sh blank    # 穴埋め版（main 用）
```

## 完成版に実行結果を埋める
`full` で生成しただけでは出力セルが空なので，GPU マシンで上から実行して埋める:
```bash
uv run jupyter nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.timeout=1200 notebooks/03_train.ipynb
```

## ブランチ運用
- `ans`: `tools/` 一式 ＋ `full` 生成＋実行済みの notebook．
- `main`: `tools/` は載せず，`blank` 生成の notebook を置く（受講者が穴埋め）．
  main の穴埋め notebook を作り直すときは `ans` で `bash tools/regen.sh blank` してから main へ反映する．
