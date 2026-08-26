#!/bin/bash
# run_1shot_remaining.sh
# 跑剩餘 7 個 1-shot 實驗（EuroSAT FSR + 其他 3 個資料集 Baseline+FSR）
# 前提：EuroSAT Baseline 已由 task-38 跑完
# 2026-08-13

set -e
cd "$(dirname "$0")"

PYTHON=/home/user/anaconda3/envs/research/bin/python
LOG_DIR="logs/1shot"
mkdir -p "$LOG_DIR"

echo "======================================================"
echo " CC-CDFSL 1-shot 剩餘實驗（7 個）"
echo " 開始時間: $(date '+%Y-%m-%d %H:%M:%S')"
echo "======================================================"

# ===== 1. EuroSAT FSR =====
echo ""
echo ">>> [Feature-SR] Dataset: eurosat  (1-shot)"
$PYTHON train.py \
    --dataset eurosat \
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
    2>&1 | tee "$LOG_DIR/eurosat_1shot_fsr.log"

# ===== 2-7. crop_disease / isic / chestx =====
for DS in "crop_disease" "isic" "chestx"; do
    echo ""
    echo ">>> [Baseline] Dataset: $DS  (1-shot)"
    $PYTHON train.py \
        --dataset "$DS" \
        --n_shot 1 \
        --epochs 100 \
        --episodes_per_epoch 100 \
        --eval_interval 10 \
        --n_eval_episodes 100 \
        --lr 1e-4 \
        --seed 42 \
        2>&1 | tee "$LOG_DIR/${DS}_1shot_baseline.log"

    echo ""
    echo ">>> [Feature-SR] Dataset: $DS  (1-shot)"
    $PYTHON train.py \
        --dataset "$DS" \
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
        2>&1 | tee "$LOG_DIR/${DS}_1shot_fsr.log"
done

echo ""
echo "======================================================"
echo " 全部完成！結束時間: $(date '+%Y-%m-%d %H:%M:%S')"
echo "======================================================"
