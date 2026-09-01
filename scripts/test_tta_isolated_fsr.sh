#!/bin/bash
# 同上，但用 Feature-SR + Ensemble 疊加後的最佳 checkpoint（目前最好的模型）
# 回答：在最終最佳模型上，TTA 還能不能再加分？
set -e
cd /var/tmp/cc-cdfsl

LOGDIR=logs/tta_check
mkdir -p "$LOGDIR"
PYTHON=/home/user/anaconda3/envs/research/bin/python

for ds in eurosat crop_disease isic chestx; do
  echo "[$(date +%H:%M:%S)] $ds fsr+ensemble (no TTA)..."
  $PYTHON evaluate.py --dataset "$ds" \
    --checkpoint "./checkpoints_fsr_ensemble/${ds}_5shot_fsr_best.pth" \
    --n_shot 5 --n_episodes 400 --use_prompt_ensemble \
    > "$LOGDIR/${ds}_fsrens_noTTA.log" 2>&1

  echo "[$(date +%H:%M:%S)] $ds fsr+ensemble + TTA..."
  $PYTHON evaluate.py --dataset "$ds" \
    --checkpoint "./checkpoints_fsr_ensemble/${ds}_5shot_fsr_best.pth" \
    --n_shot 5 --n_episodes 400 --use_prompt_ensemble --use_tta \
    > "$LOGDIR/${ds}_fsrens_TTA.log" 2>&1
done

echo "[$(date +%H:%M:%S)] ALL DONE FSR"
