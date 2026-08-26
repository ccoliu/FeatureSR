#!/bin/bash
# run_all_strict.sh
# 完整 Strict Few-Shot 實驗 (4 資料集 × 2 Shots (1 & 5) × 2 模式 (Baseline & FSR) = 16 個實驗)
# 2026-08-18

set -e
cd "$(dirname "$0")"

PYTHON=/home/user/anaconda3/envs/research/bin/python
LOG_DIR="logs/strict"
mkdir -p "$LOG_DIR"

DATASETS=("eurosat" "crop_disease" "isic" "chestx")
SHOTS=(1 5)

# 依論文 Table 13 設定各資料集的超參數 lambda1 (T-I-T) 和 lambda2 (I-T-I)
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
echo " CC-CDFSL + Feature-SR 完整 16 個 Strict Few-Shot 實驗"
echo " 開始時間: $(date '+%Y-%m-%d %H:%M:%S')"
echo "======================================================"

for SHOT in "${SHOTS[@]}"; do
    N_EVAL_EPISODES=100
    if [ "$SHOT" -eq 5 ]; then
        N_EVAL_EPISODES=400
    fi

    for DS in "${DATASETS[@]}"; do
        LAMBDAS=$(get_lambdas "$DS")

        # 1. Baseline (CC-CDFSL)
        echo ""
        echo ">>> [Baseline] Dataset: $DS | ${SHOT}-shot | Strict"
        $PYTHON train.py \
            --dataset "$DS" \
            --n_shot "$SHOT" \
            --strict_few_shot \
            --epochs 100 \
            --episodes_per_epoch 100 \
            --eval_interval 10 \
            --n_eval_episodes "$N_EVAL_EPISODES" \
            --lr 1e-4 \
            --seed 42 \
            $LAMBDAS \
            2>&1 | tee "$LOG_DIR/${DS}_${SHOT}shot_baseline_strict.log"

        # 2. Feature-SR (我們的改進方法)
        echo ""
        echo ">>> [Feature-SR] Dataset: $DS | ${SHOT}-shot | Strict"
        $PYTHON train.py \
            --dataset "$DS" \
            --n_shot "$SHOT" \
            --strict_few_shot \
            --epochs 100 \
            --episodes_per_epoch 100 \
            --eval_interval 10 \
            --n_eval_episodes "$N_EVAL_EPISODES" \
            --lr 1e-4 \
            --seed 42 \
            --use_feature_sr \
            --sr_scale 2 \
            --sr_refiner_layers 2 \
            --sr_refiner_heads 8 \
            --lambda3 0.3 \
            $LAMBDAS \
            2>&1 | tee "$LOG_DIR/${DS}_${SHOT}shot_fsr_strict.log"
    done
done

echo ""
echo "======================================================"
echo " 全部 16 個實驗完成！結束時間: $(date '+%Y-%m-%d %H:%M:%S')"
echo "======================================================"
