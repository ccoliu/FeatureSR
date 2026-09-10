#!/bin/bash
# run_1shot_full.sh
# 補齊 5-way 1-shot 的完整實驗矩陣：Baseline / Ensemble-only / Feature-SR+Ensemble
# 沿用跟 5-shot ensemble pilot 一致的協定（strict_few_shot、lr=3e-5、warmup=5、per-dataset lambda1/2、lambda3=0.3），
# 讓 1-shot 的三組結果彼此可比（同一協定下訓練），跟舊版 baseline-only 1-shot（run_1shot.sh，不同協定）分開看。
set -e
cd "$(dirname "$0")/.."

PYTHON=/home/user/anaconda3/envs/research/bin/python
LOG_DIR="logs/1shot_v2"
mkdir -p "$LOG_DIR"

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

COMMON="--n_shot 1 --strict_few_shot --epochs 100 --episodes_per_epoch 100 --eval_interval 10 --n_eval_episodes 100 --lr 3e-5 --warmup_epochs 5 --seed 42"

echo "======================================================"
echo " 5-way 1-shot 完整矩陣：Baseline / Ensemble-only / FSR+Ensemble"
echo " 開始時間: $(date '+%Y-%m-%d %H:%M:%S')"
echo "======================================================"

for DS in "${DATASETS[@]}"; do
    LAMBDAS=$(get_lambdas "$DS")

    echo ""
    echo ">>> [Baseline] $DS (1-shot, current protocol)"
    $PYTHON train.py --dataset "$DS" $COMMON $LAMBDAS \
        --save_dir "./checkpoints_1shot_baseline" \
        2>&1 | tee "$LOG_DIR/${DS}_1shot_baseline.log"

    echo ""
    echo ">>> [Ensemble-only] $DS (1-shot)"
    $PYTHON train.py --dataset "$DS" $COMMON $LAMBDAS \
        --use_prompt_ensemble \
        --save_dir "./checkpoints_1shot_ensemble" \
        2>&1 | tee "$LOG_DIR/${DS}_1shot_ensemble.log"

    echo ""
    echo ">>> [Feature-SR + Ensemble] $DS (1-shot)"
    $PYTHON train.py --dataset "$DS" $COMMON $LAMBDAS \
        --use_feature_sr --sr_scale 2 --sr_refiner_layers 2 --sr_refiner_heads 8 --lambda3 0.3 \
        --use_prompt_ensemble \
        --save_dir "./checkpoints_1shot_fsr_ensemble" \
        2>&1 | tee "$LOG_DIR/${DS}_1shot_fsr_ensemble.log"
done

echo ""
echo "======================================================"
echo " 全部完成！結束時間: $(date '+%Y-%m-%d %H:%M:%S')"
echo "======================================================"
