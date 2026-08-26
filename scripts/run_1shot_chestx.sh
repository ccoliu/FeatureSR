#!/bin/bash
# run_1shot_chestx.sh — 重跑 ChestX 1-shot (Baseline + FSR)
set -e
cd "$(dirname "$0")"

PYTHON=/home/user/anaconda3/envs/research/bin/python
LOG_DIR="logs/1shot"
mkdir -p "$LOG_DIR"

echo ">>> [Baseline] Dataset: chestx  (1-shot)"
$PYTHON train.py \
    --dataset chestx \
    --n_shot 1 \
    --epochs 100 \
    --episodes_per_epoch 100 \
    --eval_interval 10 \
    --n_eval_episodes 100 \
    --lr 1e-4 \
    --seed 42 \
    2>&1 | tee "$LOG_DIR/chestx_1shot_baseline.log"

echo ""
echo ">>> [Feature-SR] Dataset: chestx  (1-shot)"
$PYTHON train.py \
    --dataset chestx \
    --n_shot 1 \
    --epochs 100 \
    --episodes_per_epoch 100 \
    --eval_interval 10 \
    --n_eval_episodes 100 \
    --lr 1e-4 \
    --seed 42 \
    --use_feature_sr \
    --sr_scale 2 \
    --sr_refiner_layers 2 \
    --sr_refiner_heads 8 \
    --lambda3 0.3 \
    2>&1 | tee "$LOG_DIR/chestx_1shot_fsr.log"

echo "ChestX 1-shot 完成！$(date '+%Y-%m-%d %H:%M:%S')"
