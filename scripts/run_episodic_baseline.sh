#!/bin/bash
# run_episodic_baseline.sh
# 協定對齊第一階段：用 episode-level fine-tuning（CC-CDFSL / StepSPT 協定）重現純 CLIP-LoRA baseline，
# 目標是在誤差範圍內接近 CC-CDFSL 論文 Table 1 的 CLIP-LoRA (ViT/CLIP) 數字：
#   1-shot: ISIC 35.23 / ChestX 21.73 / EuroSAT 81.49 / CropDiseases 85.11
#   5-shot: ISIC 50.68 / ChestX 24.44 / EuroSAT 92.63 / CropDiseases 96.20
# 每個 episode 500 步（第一版 100 步嚴重訓練不足，見 reports/0924_Episodic協定對齊報告.md）。
# 每個 episode 結果即時寫入 results_episodic/*_steps500.jsonl，中斷後重跑同一腳本會自動續跑。
set -e
set -o pipefail
cd "$(dirname "$0")/.."

PYTHON=${PYTHON:-python}
STEPS=${STEPS:-500}
LOG_DIR="logs/episodic_baseline_steps${STEPS}"
mkdir -p "$LOG_DIR"

for SHOT in 1 5; do
  for DS in eurosat crop_disease isic chestx; do
    echo ">>> [CLIP-LoRA baseline, episodic, ${STEPS} steps] $DS ${SHOT}-shot  $(date '+%F %T')"
    $PYTHON train_episodic.py --dataset "$DS" --n_shot "$SHOT" --steps "$STEPS" \
        --save_dir ./results_episodic \
        2>&1 | tee -a "$LOG_DIR/${DS}_${SHOT}shot_cliplora.log"
  done
done
echo "全部完成 $(date '+%F %T')"
