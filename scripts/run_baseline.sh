#!/usr/bin/env bash
# ベースラインを学習 → 評価まで一気に回す再現用スクリプト．
set -euo pipefail

cd "$(dirname "$0")/.."

uv run python -m kws.train --config configs/baseline.yaml --device cuda
uv run python -m kws.evaluate --ckpt exp/baseline/best.pt --device cuda
