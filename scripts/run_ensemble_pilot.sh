#!/bin/bash
# run_ensemble_pilot.sh
# 驗證 Prompt Ensemble 訓練/測試不一致問題的修復方案：
# 訓練時就用 --use_prompt_ensemble，讓視覺端 LoRA 對齊到 ensemble 文字分布，
# 而不是像之前一樣訓練用單一 prompt、測試才臨時換成 ensemble。
#
# ISIC 5-shot pilot 已驗證：
#   Baseline (train/test 都用標準 prompt):        43.76 ± 1.38%
#   Post-hoc ensemble (訓練/測試不一致):           25.15% (-18.61%)
#   Ensemble-aware 訓練 (訓練/測試一致):           44.49 ± 0.61% (+0.73%)
#
# 這支腳本補跑另外 3 個已知 post-hoc ensemble 會掉分的資料集：
#   EuroSAT      -6.03%
#   CropDisease  -4.99%
#   ChestX       -1.60%

set -e
cd "$(dirname "$0")/.."

PYTHON=/home/user/anaconda3/envs/research/bin/python
LOG_DIR="logs/ensemble_pilot"
SAVE_DIR="./checkpoints_ensemble"
mkdir -p "$LOG_DIR" "$SAVE_DIR"

DATASETS=("eurosat" "crop_disease" "chestx")

get_lambdas() {
    local ds=$1
    case $ds in
        "eurosat") echo "--lambda1 1.5 --lambda2 0.2" ;;
        "crop_disease") echo "--lambda1 1.0 --lambda2 1.5" ;;
        "chestx") echo "--lambda1 3.0 --lambda2 0.5" ;;
    esac
}

echo "======================================================"
echo " Prompt-Ensemble-Aware 訓練驗證（EuroSAT / CropDisease / ChestX）"
echo " 開始時間: $(date '+%Y-%m-%d %H:%M:%S')"
echo "======================================================"

for DS in "${DATASETS[@]}"; do
    LAMBDAS=$(get_lambdas "$DS")
    echo ""
    echo ">>> [Ensemble-aware] Dataset: $DS | 5-shot"
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
        --use_prompt_ensemble \
        --save_dir "$SAVE_DIR" \
        2>&1 | tee "$LOG_DIR/${DS}_5shot_ensemble.log"
done

echo ""
echo "======================================================"
echo " 全部完成！結束時間: $(date '+%Y-%m-%d %H:%M:%S')"
echo "======================================================"
