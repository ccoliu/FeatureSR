#!/bin/bash
set -e
set -o pipefail
cd /home/ccoliu/FeatureSR
source .venv/bin/activate

# 等 EuroSAT λ3=0.5 訓練行程結束
until ! pgrep -f "train.py.*dataset eurosat.*lambda3 0.5" > /dev/null; do
    sleep 15
done

echo ">>> [CHAIN] EuroSAT λ3=0.5 完成，開始 ISIC λ3=0.5"
python train.py \
    --dataset isic \
    --n_shot 5 \
    --strict_few_shot \
    --epochs 100 \
    --episodes_per_epoch 100 \
    --eval_interval 10 \
    --n_eval_episodes 400 \
    --lr 1e-4 \
    --seed 42 \
    --use_feature_sr \
    --sr_scale 2 \
    --sr_refiner_layers 2 \
    --sr_refiner_heads 8 \
    --lambda3 0.5 \
    --lambda1 3.0 --lambda2 2.0 \
    --save_dir ./checkpoints_lambda3_sweep/l3_0.5 \
    2>&1 | tee logs/lambda3_sweep_4090/isic_l3_0.5.log
echo "=== ISIC λ3=0.5 完成 ==="
