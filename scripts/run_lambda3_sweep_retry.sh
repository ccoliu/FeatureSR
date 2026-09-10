#!/bin/bash
# run_lambda3_sweep_retry.sh
# 補跑 run_lambda3_sweep.sh 因 CUDA 瞬斷 / 主機重開機而遺失、且沒有任何 checkpoint 可回收的 3 組：
#   EuroSAT lambda3=0.1、EuroSAT lambda3=0.5、ISIC lambda3=0.5
# （ISIC lambda3=0.1 / lambda3=1.0 已用 epoch-10 checkpoint 補評估回收，不需重跑）
#
# 只在確認 GPU 完全閒置、且沒有其他補救評估同時進行時才啟動，避免重演 OOM。
set -e
set -o pipefail
cd "$(dirname "$0")/.."

PYTHON=/home/user/anaconda3/envs/research/bin/python
LOG_DIR="logs/lambda3_sweep"
mkdir -p "$LOG_DIR"

COMMON="--n_shot 5 --strict_few_shot --epochs 100 --episodes_per_epoch 100 \
        --eval_interval 10 --n_eval_episodes 400 --lr 1e-4 --seed 42 \
        --use_feature_sr --sr_scale 2 --sr_refiner_layers 2 --sr_refiner_heads 8"

echo "======================================================"
echo " Lambda3 sweep 補跑：EuroSAT{0.1,0.5} + ISIC{0.5}"
echo " 開始時間: $(date '+%Y-%m-%d %H:%M:%S')"
echo "======================================================"

echo ""
echo ">>> [RETRY lambda3=0.1] eurosat"
$PYTHON train.py --dataset eurosat $COMMON --lambda1 1.5 --lambda2 0.2 \
    --lambda3 0.1 --save_dir "./checkpoints_lambda3_sweep/l3_0.1" \
    2>&1 | tee "$LOG_DIR/eurosat_l3_0.1_RETRY.log"

echo ""
echo ">>> [RETRY lambda3=0.5] eurosat"
$PYTHON train.py --dataset eurosat $COMMON --lambda1 1.5 --lambda2 0.2 \
    --lambda3 0.5 --save_dir "./checkpoints_lambda3_sweep/l3_0.5" \
    2>&1 | tee "$LOG_DIR/eurosat_l3_0.5_RETRY.log"

echo ""
echo ">>> [RETRY lambda3=0.5] isic"
$PYTHON train.py --dataset isic $COMMON --lambda1 3.0 --lambda2 2.0 \
    --lambda3 0.5 --save_dir "./checkpoints_lambda3_sweep/l3_0.5" \
    2>&1 | tee "$LOG_DIR/isic_l3_0.5_RETRY.log"

echo ""
echo "======================================================"
echo " RETRY 全部完成！結束時間: $(date '+%Y-%m-%d %H:%M:%S')"
echo "======================================================"
