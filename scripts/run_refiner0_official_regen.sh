#!/bin/bash
# run_refiner0_official_regen.sh
#
# 背景：0909_FeatureSR內部Ablation報告.md 已證明 FeatureRefiner（Transformer 精煉層）
# 三個資料集上一致無貢獻（EuroSAT +0.03%、CropDiseases +0.67%、ISIC -0.17%，全落在信賴區間內），
# 且多佔 FeatureSR 新增可訓練參數的 41.6%（4.21M / 10.13M）。0912 56x56 report 進一步確認提高
# 上採樣解析度也沒有額外增益。因此「乾淨架構」= sr_refiner_layers=0, sr_scale=2, lambda3=0.3。
#
# 但目前 methodology 文件 §4.2-§4.7 的所有 FSR 相關主表數字，全部是用 refiner=2（預設值）跑的，
# 跟論文推薦的最終架構不一致。本腳本用 refiner=0 重新產生：
#   Phase 1: FSR standalone 1-shot（4 資料集）—— 5-shot 已有乾淨數字，不必重跑：
#            EuroSAT 89.96±0.43%／CropDiseases 90.23±0.65%／ISIC 43.06±0.64%（0909 ablation C組）
#            ChestX 22.70±0.46%（0912 56x56 report 的 control 組）
#   Phase 2: Feature-SR + Prompt-Ensemble 5-shot（4 資料集）
#   Phase 3: Feature-SR + Prompt-Ensemble 1-shot（4 資料集）
#   Phase 4: 用 Phase 2 產生的新 checkpoint 重新跑 TTA 有/無對照（eval-only，不需訓練）
#
# 各 Phase 的協定原封不動比照舊腳本（只把 sr_refiner_layers 從 2 改成 0）：
#   Phase 1 比照 scripts/run_all_strict.sh 的 1-shot FSR 區塊（lr=1e-4, 無 warmup, n_eval_episodes=100）
#   Phase 2 比照 scripts/run_fsr_ensemble_pilot.sh（lr=3e-5, warmup=5, n_eval_episodes=400）
#   Phase 3 比照 scripts/run_1shot_full.sh 的 FSR+Ensemble 區塊（lr=3e-5, warmup=5, n_eval_episodes=100）
#   Phase 4 比照 scripts/test_tta_isolated_fsr.sh（純 evaluate.py，無訓練）
#
# Ensemble-only（無 FSR）與 Baseline 不受 refiner 影響，不需重跑，沿用既有數字。
#
# 建議在 server127（RTX 4090）背景執行（nohup），全部跑完可能要數小時到十幾小時
# （ChestX 過去在 4090 上單組訓練就要 ~5.5 小時，是目前最大的時間變數）。
set -e
set -o pipefail
cd "$(dirname "$0")/.."

PYTHON=${PYTHON:-python}
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

echo "======================================================"
echo " Refiner=0 官方基準數字重跑（Phase 1-4）"
echo " 開始時間: $(date '+%Y-%m-%d %H:%M:%S')"
echo "======================================================"

# ─────────────────────────────────────────────
# Phase 1: FSR standalone 1-shot（refiner=0）
# ─────────────────────────────────────────────
LOG_DIR="logs/refiner0_regen/phase1_fsr_1shot"
SAVE_DIR="./checkpoints_refiner0/fsr_1shot"
mkdir -p "$LOG_DIR" "$SAVE_DIR"

for DS in "${DATASETS[@]}"; do
    LAMBDAS=$(get_lambdas "$DS")
    echo ""
    echo ">>> [Phase 1: FSR 1-shot, refiner=0] $DS"
    $PYTHON train.py --dataset "$DS" \
        --n_shot 1 --strict_few_shot --epochs 100 --episodes_per_epoch 100 \
        --eval_interval 10 --n_eval_episodes 100 --lr 1e-4 --seed 42 \
        $LAMBDAS \
        --use_feature_sr --sr_scale 2 --sr_refiner_layers 0 --sr_refiner_heads 8 --lambda3 0.3 \
        --save_dir "$SAVE_DIR" \
        2>&1 | tee "$LOG_DIR/${DS}_1shot_fsr_refiner0.log"
done

# ─────────────────────────────────────────────
# Phase 2: Feature-SR + Prompt-Ensemble 5-shot（refiner=0）
# ─────────────────────────────────────────────
LOG_DIR="logs/refiner0_regen/phase2_fsr_ensemble_5shot"
SAVE_DIR="./checkpoints_refiner0/fsr_ensemble_5shot"
mkdir -p "$LOG_DIR" "$SAVE_DIR"

for DS in "${DATASETS[@]}"; do
    LAMBDAS=$(get_lambdas "$DS")
    echo ""
    echo ">>> [Phase 2: FSR+Ensemble 5-shot, refiner=0] $DS"
    $PYTHON train.py --dataset "$DS" \
        --n_shot 5 --strict_few_shot --epochs 100 --episodes_per_epoch 100 \
        --eval_interval 10 --n_eval_episodes 400 --lr 3e-5 --warmup_epochs 5 --seed 42 \
        $LAMBDAS \
        --use_feature_sr --sr_scale 2 --sr_refiner_layers 0 --sr_refiner_heads 8 --lambda3 0.3 \
        --use_prompt_ensemble \
        --save_dir "$SAVE_DIR" \
        2>&1 | tee "$LOG_DIR/${DS}_5shot_fsr_ensemble_refiner0.log"
done

# ─────────────────────────────────────────────
# Phase 3: Feature-SR + Prompt-Ensemble 1-shot（refiner=0）
# ─────────────────────────────────────────────
LOG_DIR="logs/refiner0_regen/phase3_fsr_ensemble_1shot"
SAVE_DIR="./checkpoints_refiner0/fsr_ensemble_1shot"
mkdir -p "$LOG_DIR" "$SAVE_DIR"

for DS in "${DATASETS[@]}"; do
    LAMBDAS=$(get_lambdas "$DS")
    echo ""
    echo ">>> [Phase 3: FSR+Ensemble 1-shot, refiner=0] $DS"
    $PYTHON train.py --dataset "$DS" \
        --n_shot 1 --strict_few_shot --epochs 100 --episodes_per_epoch 100 \
        --eval_interval 10 --n_eval_episodes 100 --lr 3e-5 --warmup_epochs 5 --seed 42 \
        $LAMBDAS \
        --use_feature_sr --sr_scale 2 --sr_refiner_layers 0 --sr_refiner_heads 8 --lambda3 0.3 \
        --use_prompt_ensemble \
        --save_dir "$SAVE_DIR" \
        2>&1 | tee "$LOG_DIR/${DS}_1shot_fsr_ensemble_refiner0.log"
done

# ─────────────────────────────────────────────
# Phase 4: 用 Phase 2 的新 checkpoint 重新驗證 TTA（eval-only）
# ─────────────────────────────────────────────
LOG_DIR="logs/refiner0_regen/phase4_tta"
mkdir -p "$LOG_DIR"

for DS in "${DATASETS[@]}"; do
    echo ""
    echo ">>> [Phase 4: TTA re-check, refiner=0] $DS (no TTA)"
    $PYTHON evaluate.py --dataset "$DS" \
        --checkpoint "./checkpoints_refiner0/fsr_ensemble_5shot/${DS}_5shot_fsr_best.pth" \
        --n_shot 5 --n_episodes 400 --use_prompt_ensemble \
        > "$LOG_DIR/${DS}_fsrens_noTTA_refiner0.log" 2>&1

    echo ">>> [Phase 4: TTA re-check, refiner=0] $DS (+TTA)"
    $PYTHON evaluate.py --dataset "$DS" \
        --checkpoint "./checkpoints_refiner0/fsr_ensemble_5shot/${DS}_5shot_fsr_best.pth" \
        --n_shot 5 --n_episodes 400 --use_prompt_ensemble --use_tta \
        > "$LOG_DIR/${DS}_fsrens_TTA_refiner0.log" 2>&1
done

echo ""
echo "======================================================"
echo " Refiner=0 官方基準數字重跑全部完成！結束時間: $(date '+%Y-%m-%d %H:%M:%S')"
echo "======================================================"
