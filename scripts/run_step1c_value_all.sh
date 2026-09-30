#!/bin/bash
# run_step1c_value_all.sh
# 步驟 1c：把步驟 1b 在 ISIC 5-shot 有效的設定（value patch、分開監督 λ=1、推論 γ=1，+1.70 ± 0.53）
# 補到其餘資料集與 1-shot，確認增益是否能推廣。
# 對照組：
#   ISIC / EuroSAT → 步驟 0 的 *_steps500_cnfull.jsonl
#   CropDiseases / ChestX → 舊的 *_steps500.jsonl（兩者原本就是完整名稱，prompt 與 cnfull 完全相同）
# 便宜的 1-shot 先跑。中斷後重跑同一腳本會自動續跑。
set -e
set -o pipefail
cd "$(dirname "$0")/.."

PYTHON=${PYTHON:-.venv/bin/python}
LOG_DIR="logs/step1c_value_all"
mkdir -p "$LOG_DIR"

for RUN in "isic 1" "eurosat 1" "chestx 1" "crop_disease 1" "chestx 5" "crop_disease 5"; do
  set -- $RUN
  DS=$1; SHOT=$2
  echo ">>> [步驟 1c separate λ=1, patch=value] $DS ${SHOT}-shot  $(date '+%F %T')"
  $PYTHON train_episodic.py --dataset "$DS" --n_shot "$SHOT" --class_names full \
      --local_score --local_k_frac 0.1 --local_gamma 1.0 --patch_mode value \
      --local_loss separate --local_lambda 1.0 \
      --save_dir ./results_episodic \
      2>&1 | tee -a "$LOG_DIR/${DS}_${SHOT}shot_sep1_value.log"
done
echo "全部完成 $(date '+%F %T')"
