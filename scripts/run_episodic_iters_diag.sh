#!/bin/bash
# run_episodic_iters_diag.sh
# 診斷：episodic baseline 大幅低於論文（ISIC 5-shot 31.98 vs 50.68、CropDiseases 80.61 vs 96.20），
# 且每個 episode 結束時訓練 CE 仍接近隨機（ISIC 1.24、ChestX 1.57，ln5=1.61）⇒ 疑似訓練步數不足。
# 目前 5-shot 只有 100 步（25 張 < batch 32，1 epoch = 1 step）；CLIP-LoRA 官方是 500 × shots 步。
#
# 只跑差距最大的兩個資料集的 5-shot 前 20 個 episode。每個 episode 的亂數種子只由
# (seed, episode index) 決定，所以這 20 個 episode 與正式跑的前 20 個完全相同，可做配對比較。
#   epochs=500  ⇒ 500 步（CLIP-LoRA 1-shot 的預算）
#   epochs=2500 ⇒ 2500 步（CLIP-LoRA 5-shot 的預算 = 500 × 5）
set -e
set -o pipefail
cd "$(dirname "$0")/.."

PYTHON=${PYTHON:-python}
LOG_DIR="logs/episodic_iters_diag"
mkdir -p "$LOG_DIR"

for DS in isic crop_disease; do
  for EP in 500 2500; do
    echo ">>> [iters diag] $DS 5-shot, ${EP} steps  $(date '+%F %T')"
    $PYTHON train_episodic.py --dataset "$DS" --n_shot 5 --n_episodes 20 --epochs "$EP" \
        --log_every 5 --save_dir ./results_episodic_diag --tag "cliplora_steps${EP}" \
        2>&1 | tee -a "$LOG_DIR/${DS}_5shot_steps${EP}.log"
  done
done
echo "全部完成 $(date '+%F %T')"
