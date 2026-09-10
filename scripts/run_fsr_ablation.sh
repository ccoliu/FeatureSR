#!/bin/bash
# run_fsr_ablation.sh
# Feature-SR 模組內部 ablation：增益到底來自哪裡？
#
# 動機（見 docs/RelatedWork_FeatureUpsampling.md §2）：
#   L_CR 在結構上是 FeatUp reconstruction loss 的單視角特例，且因為只有單一視角，
#   在資訊論上無法注入新的空間細節，只能當錨定正則項。因此必須證明：
#     (a) L_CR 真的有貢獻嗎？          → 關掉它（--lambda3 0）
#     (b) 增益是不是只來自 refiner 的額外容量？ → 關掉 refiner（--sr_refiner_layers 0）
#
# 三組組態（皆開啟 --use_feature_sr，5-way 5-shot）：
#   A. full      : refiner_layers=2, lambda3=0.3  （完整 Feature-SR，內部對照基準）
#   B. no_lcr    : refiner_layers=2, lambda3=0    （拿掉 Cross-Resolution Consistency）
#   C. no_refiner: refiner_layers=0, lambda3=0.3  （只剩 bilinear + 殘差卷積 + 位置編碼）
#
# 協定嚴格對齊 scripts/run_all_strict.sh（產生 §4.2 那組 Feature-SR standalone 結果的協定）：
#   lr=1e-4、無 warmup、strict_few_shot、per-dataset λ1/λ2、seed=42、不開 prompt ensemble。
#   注意：不沿用舊有的 §4.2 數字當對照，因為 logs/strict/ 與 §4.2 表格數字對不上
#   （來自不同次 run），故 A 組必須在本批次內重跑以取得內部一致的基準。
#
# ChestX 未納入：其 Feature-SR 增益本身就在雜訊等級（+0.03%），對一個 null effect
# 做 ablation 無法產生可解讀的結論。
set -e
cd "$(dirname "$0")/.."

PYTHON=/home/user/anaconda3/envs/research/bin/python
LOG_DIR="logs/fsr_ablation"
mkdir -p "$LOG_DIR"

DATASETS=("eurosat" "crop_disease" "isic")

get_lambdas() {
    local ds=$1
    case $ds in
        "eurosat") echo "--lambda1 1.5 --lambda2 0.2" ;;
        "crop_disease") echo "--lambda1 1.0 --lambda2 1.5" ;;
        "isic") echo "--lambda1 3.0 --lambda2 2.0" ;;
    esac
}

COMMON="--n_shot 5 --strict_few_shot --epochs 100 --episodes_per_epoch 100 \
        --eval_interval 10 --n_eval_episodes 400 --lr 1e-4 --seed 42 \
        --use_feature_sr --sr_scale 2 --sr_refiner_heads 8"

echo "======================================================"
echo " Feature-SR 內部 ablation：full / no_lcr / no_refiner"
echo " 開始時間: $(date '+%Y-%m-%d %H:%M:%S')"
echo "======================================================"

for DS in "${DATASETS[@]}"; do
    LAMBDAS=$(get_lambdas "$DS")

    echo ""
    echo ">>> [A: full] $DS (refiner=2, lambda3=0.3)"
    $PYTHON train.py --dataset "$DS" $COMMON $LAMBDAS \
        --sr_refiner_layers 2 --lambda3 0.3 \
        --save_dir "./checkpoints_ablation_full" \
        2>&1 | tee "$LOG_DIR/${DS}_A_full.log"

    echo ""
    echo ">>> [B: no_lcr] $DS (refiner=2, lambda3=0)"
    $PYTHON train.py --dataset "$DS" $COMMON $LAMBDAS \
        --sr_refiner_layers 2 --lambda3 0 \
        --save_dir "./checkpoints_ablation_no_lcr" \
        2>&1 | tee "$LOG_DIR/${DS}_B_no_lcr.log"

    echo ""
    echo ">>> [C: no_refiner] $DS (refiner=0, lambda3=0.3)"
    $PYTHON train.py --dataset "$DS" $COMMON $LAMBDAS \
        --sr_refiner_layers 0 --lambda3 0.3 \
        --save_dir "./checkpoints_ablation_no_refiner" \
        2>&1 | tee "$LOG_DIR/${DS}_C_no_refiner.log"
done

echo ""
echo "======================================================"
echo " ABLATION 全部完成！結束時間: $(date '+%Y-%m-%d %H:%M:%S')"
echo "======================================================"
