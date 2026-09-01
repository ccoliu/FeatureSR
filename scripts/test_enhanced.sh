#!/bin/bash
# test_enhanced.sh
# 一鍵測試 Method 1 (領域 Multi-Prompt 集成) 與 Method 4 (測試期多視角 TTA)

set -e
cd "$(dirname "$0")/.."

PYTHON=/home/user/anaconda3/envs/research/bin/python
DATASET=${1:-"all"}
SHOT=${2:-5}
EPISODES=${3:-100}

echo "========================================================================="
echo " 🚀 FeatureSR 增強推論測試 (Prompt Ensemble M1 + Test-Time Augmentation M4)"
echo " 資料集: $DATASET | Shot: $SHOT | 測試 Episodes: $EPISODES"
echo "========================================================================="

$PYTHON experiments/evaluate_enhanced.py \
    --dataset "$DATASET" \
    --n_shot "$SHOT" \
    --n_episodes "$EPISODES" \
    --use_fsr_weights
