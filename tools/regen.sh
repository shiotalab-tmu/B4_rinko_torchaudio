#!/usr/bin/env bash
# notebooks/*.ipynb を一括生成する．
#   mode = full  … 完成版（ans ブランチ用）
#   mode = blank … 穴埋め版（main ブランチ用）
# 完成版は出力を埋めるため別途 nbconvert --execute する（README 参照）．
set -euo pipefail
cd "$(dirname "$0")/.."

MODE="${1:-full}"
for g in 01_data 02_model 03_train 04_eval_tune; do
    uv run python "tools/gen_${g}.py" "notebooks/${g}.ipynb" "$MODE"
done
echo "done (mode=${MODE})"
