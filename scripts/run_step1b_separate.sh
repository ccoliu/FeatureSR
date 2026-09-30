#!/bin/bash
# run_step1b_separate.sh
# 步驟 1b：分開監督。步驟 1（run_step1_local.sh）用融合 logits 訓練，CLS 全面退化，增益為負。
# 這裡改成 loss = CE(s·CLS·T) + λ·CE(s·local)，λ=1；推論仍用 CLS + γ·local（γ=1 事先固定）。
# 對照組是步驟 0 的 cnfull baseline（同一批 episode）。
# 優先順序：value patch 的 ISIC、EuroSAT 5-shot 先跑（約 5.5 小時），plain 排在後面，可視結果提前停止。
# 中斷後重跑同一腳本會自動續跑。
set -e
set -o pipefail
cd "$(dirname "$0")/.."

PYTHON=${PYTHON:-.venv/bin/python}
LOG_DIR="logs/step1b_separate"
mkdir -p "$LOG_DIR"

for RUN in "value isic" "value eurosat" "plain isic" "plain eurosat"; do
  set -- $RUN
  PM=$1; DS=$2
  echo ">>> [步驟 1b separate λ=1, patch=$PM] $DS 5-shot  $(date '+%F %T')"
  $PYTHON train_episodic.py --dataset "$DS" --n_shot 5 --class_names full \
      --local_score --local_k_frac 0.1 --local_gamma 1.0 --patch_mode "$PM" \
      --local_loss separate --local_lambda 1.0 \
      --save_dir ./results_episodic \
      2>&1 | tee -a "$LOG_DIR/${DS}_5shot_sep1_${PM}.log"
done
echo "全部完成 $(date '+%F %T')"
