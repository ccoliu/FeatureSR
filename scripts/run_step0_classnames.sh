#!/bin/bash
# run_step0_classnames.sh
# 步驟 0（reports/0928_交接與下一步.md）：prompt 改用完整類別名稱（--class_names full，同 ATHA），
# 重跑純 CLIP-LoRA baseline。舊的縮寫名稱結果（results_episodic/*_steps500.jsonl）就是 raw 對照組，
# 同一批 episode，可直接配對：
#   python scripts/paired_diff.py results_episodic/isic_5shot_nofsr_nocc_loraboth_r2_steps500{_cnfull,}.jsonl
# 驗證標準：ISIC 5-shot 從 47.71 升到約 50.7–51.8（論文 CLIP-LoRA 50.68；ATHA 程式碼 51.83）。
# ISIC 5-shot 排第一個，跑完就先看。中斷後重跑同一腳本會自動續跑。
set -e
set -o pipefail
cd "$(dirname "$0")/.."

PYTHON=${PYTHON:-.venv/bin/python}
LOG_DIR="logs/step0_classnames"
mkdir -p "$LOG_DIR"

for RUN in "isic 5" "eurosat 5" "isic 1" "eurosat 1"; do
  set -- $RUN
  DS=$1; SHOT=$2
  echo ">>> [步驟 0 CLIP-LoRA baseline, cnfull] $DS ${SHOT}-shot  $(date '+%F %T')"
  $PYTHON train_episodic.py --dataset "$DS" --n_shot "$SHOT" --class_names full \
      --save_dir ./results_episodic \
      2>&1 | tee -a "$LOG_DIR/${DS}_${SHOT}shot_cliplora_cnfull.log"
done
echo "全部完成 $(date '+%F %T')"
