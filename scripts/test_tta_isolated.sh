#!/bin/bash
# 隔離測試 TTA (水平翻轉多視角) 在「已修復的 ensemble-aware checkpoint」上的真實效果
# 對照組：--use_prompt_ensemble (無 TTA，重跑一次確認可重現)
# 實驗組：--use_prompt_ensemble --use_tta
set -e
cd /var/tmp/cc-cdfsl

LOGDIR=logs/tta_check
mkdir -p "$LOGDIR"
PYTHON=/home/user/anaconda3/envs/research/bin/python

for ds in eurosat crop_disease isic chestx; do
  echo "[$(date +%H:%M:%S)] $ds ensemble-only (no TTA)..."
  $PYTHON evaluate.py --dataset "$ds" \
    --checkpoint "./checkpoints_ensemble/${ds}_5shot_best.pth" \
    --n_shot 5 --n_episodes 400 --use_prompt_ensemble \
    > "$LOGDIR/${ds}_ens_noTTA.log" 2>&1

  echo "[$(date +%H:%M:%S)] $ds ensemble + TTA..."
  $PYTHON evaluate.py --dataset "$ds" \
    --checkpoint "./checkpoints_ensemble/${ds}_5shot_best.pth" \
    --n_shot 5 --n_episodes 400 --use_prompt_ensemble --use_tta \
    > "$LOGDIR/${ds}_ens_TTA.log" 2>&1
done

echo "[$(date +%H:%M:%S)] ALL DONE"
