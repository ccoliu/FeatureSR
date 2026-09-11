#!/bin/bash
# run_lambda3_sweep_eurosat_high.sh
# EuroSAT 在 lambda3=1.0 時仍呈單調上升、未見頂點，往上加測 {1.5, 2.0} 以逼近真正的峰值/拐點。
# 只測 EuroSAT：ISIC 已在 {0, 0.1, 0.3, 0.5, 1.0} 呈現明確倒 U 型峰值（0.3），不需再往上加測。
# 協定與 scripts/run_lambda3_sweep.sh 完全一致（lambda1/lambda2 固定於 Table 13 值，
# refiner_layers=2，strict_few_shot，5-way 5-shot，400 episodes）。
# 建議在 server127（RTX 4090）執行：兩組獨立、互不依賴，GPU 記憶體充裕時可各開一個
# terminal/tmux pane 平行跑，互相不影響。
set -e
set -o pipefail
cd "$(dirname "$0")/.."

PYTHON=${PYTHON:-python}
LOG_DIR="logs/lambda3_sweep"
mkdir -p "$LOG_DIR"

COMMON="--n_shot 5 --strict_few_shot --epochs 100 --episodes_per_epoch 100 \
        --eval_interval 10 --n_eval_episodes 400 --lr 1e-4 --seed 42 \
        --use_feature_sr --sr_scale 2 --sr_refiner_layers 2 --sr_refiner_heads 8"

echo "======================================================"
echo " EuroSAT lambda3 高值加測：{1.5, 2.0}"
echo " 開始時間: $(date '+%Y-%m-%d %H:%M:%S')"
echo "======================================================"

echo ""
echo ">>> [lambda3=1.5] eurosat"
$PYTHON train.py --dataset eurosat $COMMON --lambda1 1.5 --lambda2 0.2 \
    --lambda3 1.5 --save_dir "./checkpoints_lambda3_sweep/l3_1.5" \
    2>&1 | tee "$LOG_DIR/eurosat_l3_1.5.log"

echo ""
echo ">>> [lambda3=2.0] eurosat"
$PYTHON train.py --dataset eurosat $COMMON --lambda1 1.5 --lambda2 0.2 \
    --lambda3 2.0 --save_dir "./checkpoints_lambda3_sweep/l3_2.0" \
    2>&1 | tee "$LOG_DIR/eurosat_l3_2.0.log"

echo ""
echo "======================================================"
echo " EuroSAT 高值加測全部完成！結束時間: $(date '+%Y-%m-%d %H:%M:%S')"
echo "======================================================"
