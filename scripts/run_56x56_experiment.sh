#!/bin/bash
# run_56x56_experiment.sh
# 測試將 Feature-SR 的上採樣網格從 28x28 (scale=2) 提高到 56x56 (scale=4)，
# 是否能提供更細緻的局部特徵、進一步提升分類準確率。
#
# 設計決策：
#   --sr_refiner_layers 0：0909 ablation report 已證明 FeatureRefiner 三個資料集上一致無貢獻，
#     且 Refiner 的 self-attention 是 O(P^2)，56x56=3136 個 patch 會讓其計算量比 28x28 暴增 16 倍；
#     關掉它既符合已驗證的架構建議，也避開不必要的平方成本。Cycle Consistency Loss 本身
#     對 patch 數只是線性 (O(P))，56x56 不會有量級上的爆炸。
#   --lambda3 0.3：沿用論文預設值，不與解析度這個變因混在一起（0910 report 建議維持統一 0.3）。
#   四個資料集全測（含 CropDiseases / ChestX）：CropDiseases 對 CR loss/refiner 不敏感，
#     但解析度提升是不同的變因（更細的病斑紋理細節），不能完全排除有效可能；ChestX 的
#     Feature-SR 整體效果雖是雜訊等級，一併測出來也能在報告中明確交代「連解析度提升都救不了」。
#
# 對照組：scripts/run_fsr_ablation.sh 的 C(no_refiner) 組（scale=2, refiner=0, lambda3=0.3），
# 三個資料集（不含 ChestX）已有結果：EuroSAT 89.96%, CropDiseases 90.23%, ISIC 43.06%。
# ChestX 的對照組本次一併補測（原 ablation 未納入 ChestX）。
set -e
set -o pipefail
cd "$(dirname "$0")/.."

PYTHON=${PYTHON:-python}
LOG_DIR="logs/56x56_experiment"
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

COMMON="--n_shot 5 --strict_few_shot --epochs 100 --episodes_per_epoch 100 \
        --eval_interval 10 --n_eval_episodes 400 --lr 1e-4 --seed 42 \
        --use_feature_sr --sr_scale 4 --sr_refiner_layers 0 --sr_refiner_heads 8 \
        --lambda3 0.3"

echo "======================================================"
echo " 56x56 Feature-SR 解析度實驗（4 資料集）"
echo " 開始時間: $(date '+%Y-%m-%d %H:%M:%S')"
echo "======================================================"

for DS in "${DATASETS[@]}"; do
    LAMBDAS=$(get_lambdas "$DS")
    echo ""
    echo ">>> [56x56] $DS"
    $PYTHON train.py --dataset "$DS" $COMMON $LAMBDAS \
        --save_dir "./checkpoints_56x56/${DS}" \
        2>&1 | tee "$LOG_DIR/${DS}_56x56.log"
done

echo ""
echo "======================================================"
echo " 56x56 實驗全部完成！結束時間: $(date '+%Y-%m-%d %H:%M:%S')"
echo "======================================================"

# 順便補跑 ChestX 的 scale=2/refiner=0 對照組（原 0909 ablation 未納入 ChestX）
echo ""
echo ">>> [對照組 28x28, no_refiner] chestx"
$PYTHON train.py --dataset chestx --n_shot 5 --strict_few_shot --epochs 100 --episodes_per_epoch 100 \
    --eval_interval 10 --n_eval_episodes 400 --lr 1e-4 --seed 42 \
    --use_feature_sr --sr_scale 2 --sr_refiner_layers 0 --sr_refiner_heads 8 \
    --lambda3 0.3 --lambda1 3.0 --lambda2 0.5 \
    --save_dir "./checkpoints_56x56/chestx_28x28_control" \
    2>&1 | tee "$LOG_DIR/chestx_28x28_control.log"

echo "全部（含對照組）完成！結束時間: $(date '+%Y-%m-%d %H:%M:%S')"
