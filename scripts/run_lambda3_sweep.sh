#!/bin/bash
# run_lambda3_sweep.sh
# lambda3 (L_CR 權重) 的一維敏感度分析（控制變因：lambda1/lambda2 固定於論文 Table 13 值，
# refiner_layers 固定為 2，與 scripts/run_fsr_ablation.sh 的 A/B 組協定完全一致）。
#
# 動機：回應教授「grid search 怎麼找」的提問。三維 (lambda1, lambda2, lambda3) 聯合搜尋
# 在單張 3060 上不可行（5^3 組合 x 2 資料集 x ~6hr/run 遠超算力）；lambda1/lambda2 沿用
# CC-CDFSL 論文 Table 13 的既有驗證值，只對本工作自己引入、從未驗證過的 lambda3 做
# one-variable-at-a-time 敏感度分析，並在論文中明確記錄這個簡化。
#
# 只掃 EuroSAT / ISIC：ablation 報告（reports/0909_FeatureSR內部Ablation報告.md）已證明
# L_CR 只在這兩個資料集上有真實貢獻，CropDiseases 對 L_CR 不敏感、ChestX 的 Feature-SR
# 效果本身就是雜訊等級，兩者都不值得花算力做敏感度分析。
#
# lambda3=0 與 lambda3=0.3 兩點已由 run_fsr_ablation.sh 的 B(no_lcr)/A(full) 組跑過，
# 此腳本只補 {0.1, 0.5, 1.0} 三個新值。
set -e
set -o pipefail
cd "$(dirname "$0")/.."

PYTHON=/home/user/anaconda3/envs/research/bin/python
LOG_DIR="logs/lambda3_sweep"
mkdir -p "$LOG_DIR"

DATASETS=("eurosat" "isic")
LAMBDA3_VALUES=(0.1 0.5 1.0)

get_lambdas() {
    local ds=$1
    case $ds in
        "eurosat") echo "--lambda1 1.5 --lambda2 0.2" ;;
        "isic") echo "--lambda1 3.0 --lambda2 2.0" ;;
    esac
}

COMMON="--n_shot 5 --strict_few_shot --epochs 100 --episodes_per_epoch 100 \
        --eval_interval 10 --n_eval_episodes 400 --lr 1e-4 --seed 42 \
        --use_feature_sr --sr_scale 2 --sr_refiner_layers 2 --sr_refiner_heads 8"

echo "======================================================"
echo " Lambda3 敏感度分析：{0.1, 0.5, 1.0} x {EuroSAT, ISIC}"
echo " （0 與 0.3 兩點已由 run_fsr_ablation.sh 的 B/A 組涵蓋）"
echo " 開始時間: $(date '+%Y-%m-%d %H:%M:%S')"
echo "======================================================"

for DS in "${DATASETS[@]}"; do
    LAMBDAS=$(get_lambdas "$DS")

    for L3 in "${LAMBDA3_VALUES[@]}"; do
        echo ""
        echo ">>> [lambda3=$L3] $DS"
        $PYTHON train.py --dataset "$DS" $COMMON $LAMBDAS \
            --lambda3 "$L3" \
            --save_dir "./checkpoints_lambda3_sweep/l3_${L3}" \
            2>&1 | tee "$LOG_DIR/${DS}_l3_${L3}.log"
    done
done

echo ""
echo "======================================================"
echo " LAMBDA3 SWEEP 全部完成！結束時間: $(date '+%Y-%m-%d %H:%M:%S')"
echo "======================================================"
