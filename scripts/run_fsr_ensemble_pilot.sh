#!/bin/bash
# run_fsr_ensemble_pilot.sh
# 驗證 Feature-SR (28x28) 與 Prompt-Ensemble-Aware 訓練是否可疊加增益。
# 兩者已分別驗證有效：
#   Feature-SR 平均增益：+1.02% (reports/0826成果報告.md)
#   Ensemble-aware 訓練平均增益：+1.21% (reports/0829_PromptEnsemble修復報告.md)
# 本腳本同時開啟 --use_feature_sr 與 --use_prompt_ensemble，跑完整 4 個資料集 5-shot。

set -e
cd "$(dirname "$0")/.."

PYTHON=/home/user/anaconda3/envs/research/bin/python
LOG_DIR="logs/fsr_ensemble_pilot"
SAVE_DIR="./checkpoints_fsr_ensemble"
mkdir -p "$LOG_DIR" "$SAVE_DIR"

DATASETS=("eurosat" "crop_disease" "isic" "chestx")

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
echo " Feature-SR + Prompt-Ensemble-Aware 疊加驗證（4 個資料集 5-shot）"
echo " 開始時間: $(date '+%Y-%m-%d %H:%M:%S')"
echo "======================================================"

for DS in "${DATASETS[@]}"; do
    LAMBDAS=$(get_lambdas "$DS")
    echo ""
    echo ">>> [Feature-SR + Ensemble] Dataset: $DS | 5-shot"
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
        --use_feature_sr \
        --sr_scale 2 \
        --sr_refiner_layers 2 \
        --sr_refiner_heads 8 \
        --lambda3 0.3 \
        --use_prompt_ensemble \
        --save_dir "$SAVE_DIR" \
        2>&1 | tee "$LOG_DIR/${DS}_5shot_fsr_ensemble.log"
done

echo ""
echo "======================================================"
echo " 全部完成！結束時間: $(date '+%Y-%m-%d %H:%M:%S')"
echo "======================================================"
