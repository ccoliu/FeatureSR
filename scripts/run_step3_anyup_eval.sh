#!/bin/bash
# run_step3_anyup_eval.sh
# 步驟 3（推論時版本）：訓練用目前最強的設定（value patch、分開監督 λ=1、γ=1、14×14 + 3×3 平滑），
# 推論時在同一個模型上另外用凍結的 AnyUp（28×28、56×56）與 bilinear（28×28）上採樣 14×14 patch 計算 local 分數。
# 主指標（acc）與 run_step2c 的平滑 14×14 結果逐 episode 相同（已驗證），AnyUp 結果在 acc_any28 / acc_any56 等欄位，
# 用 scripts/within_model_diff.py 做同模型配對比較。
# ChestX、CropDiseases 的平滑 14×14 也是第一次跑（ISIC 已在 run_step2c）。中斷後重跑同一腳本會自動續跑。
set -e
set -o pipefail
cd "$(dirname "$0")/.."

PYTHON=${PYTHON:-.venv/bin/python}
LOG_DIR="logs/step3_anyup_eval"
mkdir -p "$LOG_DIR"

for DS in isic chestx crop_disease; do
  echo ">>> [步驟 3 推論時 AnyUp 診斷] $DS 5-shot  $(date '+%F %T')"
  $PYTHON train_episodic.py --dataset "$DS" --n_shot 5 --class_names full \
      --local_score --local_k_frac 0.1 --local_gamma 1.0 --patch_mode value \
      --local_loss separate --local_lambda 1.0 --patch_smooth 3 --eval_anyup \
      --save_dir ./results_episodic \
      2>&1 | tee -a "$LOG_DIR/${DS}_5shot_anyup_eval.log"
done
echo "全部完成 $(date '+%F %T')"
