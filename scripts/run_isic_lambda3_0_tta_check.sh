#!/bin/bash
# run_isic_lambda3_0_tta_check.sh
#
# 隔離驗證 §6.2 的開放假設：ISIC 在 Feature-SR+Ensemble 疊加 TTA 時的退步
# （refiner=0 下已縮小為 -0.85%，不再顯著，但方向未變）是否是 lambda3
# （Cross-Resolution Consistency Loss 權重）本身造成的。
#
# 方法：只把 lambda3 從論文預設值 0.3 改成 0，其餘完全比照
# scripts/run_refiner0_official_regen.sh 的 Phase 2（ISIC 5-shot FSR+Ensemble）協定，
# 重新訓練一組，再用同一支 checkpoint 重新做一次 TTA 有/無對照。
#
# 判讀方式：
#   若 lambda3=0 時 TTA 的 Delta 明顯縮小或轉正 → 支持 L_CR 是退步主因的假設
#   若 lambda3=0 時 TTA 的 Delta 依然是相近幅度的負值 → 退步另有原因（可能是 ISIC
#     資料域本身對水平翻轉敏感，或是 Ensemble/LoRA 校準交互作用），與 L_CR 無關
set -e
set -o pipefail
cd "$(dirname "$0")/.."

PYTHON=${PYTHON:-python}
LOG_DIR="logs/isic_lambda3_0_tta_check"
SAVE_DIR="./checkpoints_refiner0/fsr_ensemble_5shot_lambda3_0"
mkdir -p "$LOG_DIR" "$SAVE_DIR"

echo "======================================================"
echo " ISIC lambda3=0 TTA 隔離驗證"
echo " 開始時間: $(date '+%Y-%m-%d %H:%M:%S')"
echo "======================================================"

echo ""
echo ">>> [訓練] ISIC FSR+Ensemble 5-shot, refiner=0, lambda3=0"
$PYTHON train.py --dataset isic \
    --n_shot 5 --strict_few_shot --epochs 100 --episodes_per_epoch 100 \
    --eval_interval 10 --n_eval_episodes 400 --lr 3e-5 --warmup_epochs 5 --seed 42 \
    --lambda1 3.0 --lambda2 2.0 \
    --use_feature_sr --sr_scale 2 --sr_refiner_layers 0 --sr_refiner_heads 8 --lambda3 0 \
    --use_prompt_ensemble \
    --save_dir "$SAVE_DIR" \
    2>&1 | tee "$LOG_DIR/isic_5shot_fsr_ensemble_lambda3_0.log"

echo ""
echo ">>> [評估] 無 TTA"
$PYTHON evaluate.py --dataset isic \
    --checkpoint "$SAVE_DIR/isic_5shot_fsr_best.pth" \
    --n_shot 5 --n_episodes 400 --use_prompt_ensemble \
    > "$LOG_DIR/isic_fsrens_noTTA_lambda3_0.log" 2>&1

echo ">>> [評估] +TTA"
$PYTHON evaluate.py --dataset isic \
    --checkpoint "$SAVE_DIR/isic_5shot_fsr_best.pth" \
    --n_shot 5 --n_episodes 400 --use_prompt_ensemble --use_tta \
    > "$LOG_DIR/isic_fsrens_TTA_lambda3_0.log" 2>&1

echo ""
echo "======================================================"
echo " 完成！結束時間: $(date '+%Y-%m-%d %H:%M:%S')"
echo "======================================================"
