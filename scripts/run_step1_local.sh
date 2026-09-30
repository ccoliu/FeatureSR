#!/bin/bash
# run_step1_local.sh
# 步驟 1（reports/0928_交接與下一步.md）：patch 層級分數進入分類路徑，不做 SR。
#   logits = s·(CLS·T + γ·local)，local_c = top-k（10%）patch 相似度平均，γ=1 固定
# 兩種 patch：plain（最後一層完整輸出）、value（MaskCLIP 式 value→out_proj）。
# 對照組是步驟 0 的 cnfull baseline（同一批 episode）：
#   python scripts/paired_diff.py results_episodic/isic_5shot_nofsr_nocc_loraboth_r2_steps500_cnfull_loc0.1g1.jsonl \
#                                 results_episodic/isic_5shot_nofsr_nocc_loraboth_r2_steps500_cnfull.jsonl
# 篩選點：若兩種 patch 在 ISIC、EuroSAT 5-shot 的配對 CI 都含 0，就停在步驟 1。
# ISIC 5-shot 排最前面，跑完就先看。中斷後重跑同一腳本會自動續跑。
set -e
set -o pipefail
cd "$(dirname "$0")/.."

PYTHON=${PYTHON:-.venv/bin/python}
LOG_DIR="logs/step1_local"
mkdir -p "$LOG_DIR"

for RUN in "isic 5" "eurosat 5" "isic 1" "eurosat 1"; do
  set -- $RUN
  DS=$1; SHOT=$2
  for PM in plain value; do
    echo ">>> [步驟 1 local score, patch=$PM] $DS ${SHOT}-shot  $(date '+%F %T')"
    $PYTHON train_episodic.py --dataset "$DS" --n_shot "$SHOT" --class_names full \
        --local_score --local_k_frac 0.1 --local_gamma 1.0 --patch_mode "$PM" \
        --save_dir ./results_episodic \
        2>&1 | tee -a "$LOG_DIR/${DS}_${SHOT}shot_local_${PM}.log"
  done
done
echo "全部完成 $(date '+%F %T')"
