#!/bin/bash
# run_episodic_stage2.sh
# 協定對齊第二階段（見 reports/0924_Episodic協定對齊報告.md）：
#   A. +CC-CDFSL（λ1/λ2 = 論文 Table 13）：檢驗能否重現論文報告的相對增益
#        1-shot: ISIC +2.90 / ChestX +0.48 / EuroSAT +4.58 / CropDiseases +3.80
#        5-shot: ISIC +4.04 / ChestX +1.03 / EuroSAT +1.72 / CropDiseases +0.88
#   B. +CC-CDFSL + FeatureSR（refiner=0, λ3=0.3，§6.3 的正式架構）：本論文貢獻的真正檢驗
# 與 baseline（run_episodic_baseline.sh）同為每個 episode 500 步、同一組 episode 種子 ⇒ 三者可逐 episode 配對比較。
# 順序：先跑便宜的 1-shot（約 5 小時看到第一批完整結果），再跑 5-shot，最關鍵的 ISIC / EuroSAT 優先。
# 中斷後重跑同一腳本會從斷點續跑。
set -e
set -o pipefail
cd "$(dirname "$0")/.."

PYTHON=${PYTHON:-python}
STEPS=${STEPS:-500}
LOG_DIR="logs/episodic_stage2_steps${STEPS}"
mkdir -p "$LOG_DIR"

get_lambdas() {
    case $1 in
        "eurosat") echo "--lambda1 1.5 --lambda2 0.2" ;;
        "crop_disease") echo "--lambda1 1.0 --lambda2 1.5" ;;
        "isic") echo "--lambda1 3.0 --lambda2 2.0" ;;
        "chestx") echo "--lambda1 3.0 --lambda2 0.5" ;;
    esac
}

run_pair() {
    local DS=$1 SHOT=$2
    local LAMBDAS
    LAMBDAS=$(get_lambdas "$DS")
    echo ">>> [+CC-CDFSL] $DS ${SHOT}-shot  $(date '+%F %T')"
    $PYTHON train_episodic.py --dataset "$DS" --n_shot "$SHOT" --steps "$STEPS" $LAMBDAS \
        --save_dir ./results_episodic 2>&1 | tee -a "$LOG_DIR/${DS}_${SHOT}shot_cc.log"
    echo ">>> [+CC-CDFSL +FeatureSR] $DS ${SHOT}-shot  $(date '+%F %T')"
    $PYTHON train_episodic.py --dataset "$DS" --n_shot "$SHOT" --steps "$STEPS" $LAMBDAS \
        --use_feature_sr --sr_scale 2 --sr_refiner_layers 0 --lambda3 0.3 \
        --save_dir ./results_episodic 2>&1 | tee -a "$LOG_DIR/${DS}_${SHOT}shot_cc_fsr.log"
}

for DS in isic eurosat crop_disease chestx; do run_pair "$DS" 1; done
for DS in isic eurosat crop_disease chestx; do run_pair "$DS" 5; done
echo "全部完成 $(date '+%F %T')"
