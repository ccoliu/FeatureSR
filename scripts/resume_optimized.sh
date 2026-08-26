#!/bin/bash
# resume_optimized.sh
# 續跑未完成的優化版 Strict Few-Shot 實驗
set -e
cd "$(dirname "$0")/.."

PYTHON=/home/user/anaconda3/envs/research/bin/python
LOG_DIR="logs/optimized"
mkdir -p "$LOG_DIR"

get_lambdas() {
    local ds=$1
    case $ds in
        "eurosat") echo "--lambda1 1.5 --lambda2 0.2" ;;
        "crop_disease") echo "--lambda1 1.0 --lambda2 1.5" ;;
        "isic") echo "--lambda1 3.0 --lambda2 2.0" ;;
        "chestx") echo "--lambda1 3.0 --lambda2 0.5" ;;
    esac
}

echo "======================================================"
echo " 續跑優化版 Strict Few-Shot 實驗 (lr=3e-5, warmup=5)"
echo " 開始時間: $(date '+%Y-%m-%d %H:%M:%S')"
echo "======================================================"

# 1. 補跑中斷的 ISIC 1-shot FSR
echo ""
echo ">>> [Feature-SR] Dataset: isic | 1-shot | Strict (Optimized LR/Warmup)"
$PYTHON train.py \
    --dataset isic \
    --n_shot 1 \
    --strict_few_shot \
    --epochs 100 \
    --episodes_per_epoch 100 \
    --eval_interval 10 \
    --n_eval_episodes 100 \
    --lr 3e-5 \
    --warmup_epochs 5 \
    --seed 42 \
    --use_feature_sr \
    --sr_scale 2 \
    --sr_refiner_layers 2 \
    --sr_refiner_heads 8 \
    --lambda3 0.3 \
    $(get_lambdas isic) \
    2>&1 | tee "$LOG_DIR/isic_1shot_fsr_opt.log"

# 2. ChestX 1-shot Baseline
echo ""
echo ">>> [Baseline] Dataset: chestx | 1-shot | Strict (Optimized LR/Warmup)"
$PYTHON train.py \
    --dataset chestx \
    --n_shot 1 \
    --strict_few_shot \
    --epochs 100 \
    --episodes_per_epoch 100 \
    --eval_interval 10 \
    --n_eval_episodes 100 \
    --lr 3e-5 \
    --warmup_epochs 5 \
    --seed 42 \
    $(get_lambdas chestx) \
    2>&1 | tee "$LOG_DIR/chestx_1shot_baseline_opt.log"

# 3. ChestX 1-shot FSR
echo ""
echo ">>> [Feature-SR] Dataset: chestx | 1-shot | Strict (Optimized LR/Warmup)"
$PYTHON train.py \
    --dataset chestx \
    --n_shot 1 \
    --strict_few_shot \
    --epochs 100 \
    --episodes_per_epoch 100 \
    --eval_interval 10 \
    --n_eval_episodes 100 \
    --lr 3e-5 \
    --warmup_epochs 5 \
    --seed 42 \
    --use_feature_sr \
    --sr_scale 2 \
    --sr_refiner_layers 2 \
    --sr_refiner_heads 8 \
    --lambda3 0.3 \
    $(get_lambdas chestx) \
    2>&1 | tee "$LOG_DIR/chestx_1shot_fsr_opt.log"

# 4. 全部 5-shot 實驗 (8 個)
DATASETS=("eurosat" "crop_disease" "isic" "chestx")
for DS in "${DATASETS[@]}"; do
    LAMBDAS=$(get_lambdas "$DS")
    echo ""
    echo ">>> [Baseline] Dataset: $DS | 5-shot | Strict (Optimized LR/Warmup)"
    $PYTHON train.py \
        --dataset "$DS" \
        --n_shot 5 \
        --strict_few_shot \
        --epochs 100 \
        --episodes_per_epoch 100 \
        --eval_interval 10 \
        --n_eval_episodes 400 \
        --lr 3e-5 \
        --warmup_epochs 5 \
        --seed 42 \
        $LAMBDAS \
        2>&1 | tee "$LOG_DIR/${DS}_5shot_baseline_opt.log"

    echo ""
    echo ">>> [Feature-SR] Dataset: $DS | 5-shot | Strict (Optimized LR/Warmup)"
    $PYTHON train.py \
        --dataset "$DS" \
        --n_shot 5 \
        --strict_few_shot \
        --epochs 100 \
        --episodes_per_epoch 100 \
        --eval_interval 10 \
        --n_eval_episodes 400 \
        --lr 3e-5 \
        --warmup_epochs 5 \
        --seed 42 \
        --use_feature_sr \
        --sr_scale 2 \
        --sr_refiner_layers 2 \
        --sr_refiner_heads 8 \
        --lambda3 0.3 \
        $LAMBDAS \
        2>&1 | tee "$LOG_DIR/${DS}_5shot_fsr_opt.log"
done

echo ""
echo "======================================================"
echo " 剩餘實驗全數完成！結束時間: $(date '+%Y-%m-%d %H:%M:%S')"
echo "======================================================"
