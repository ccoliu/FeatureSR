#!/bin/bash
# run_step2_fsr.sh
# 步驟 2（reports/0928_交接與下一步.md、reports/0930_步驟1）：FeatureSR 進入分類路徑。
# 設定同步驟 1 的最終設定（value patch、分開監督 λ=1、推論 γ=1），patch 改用 FeatureSR（refiner=0、scale=2）
# 上採樣到 28×28 後的特徵做 top-k（k = 10% × 784 = 78），λ3 = 0.3 / 0 各一組。
# 對照組是步驟 1 的 14×14 結果（同一批 episode），回答「上採樣本身有沒有增益」：
#   python scripts/paired_diff.py results_episodic/isic_5shot_fsr_s2_ref0_l30.3_nocc_loraboth_r2_steps500_cnfull_loc0.1g1_sep1_pmvalue.jsonl \
#                                 results_episodic/isic_5shot_nofsr_nocc_loraboth_r2_steps500_cnfull_loc0.1g1_sep1_pmvalue.jsonl
# 只跑步驟 1 有增益的三個資料集的 5-shot。4090 上約 29 秒/episode，每組約 3.2 小時。
# 中斷後重跑同一腳本會自動續跑。
set -e
set -o pipefail
cd "$(dirname "$0")/.."

PYTHON=${PYTHON:-.venv/bin/python}
LOG_DIR="logs/step2_fsr"
mkdir -p "$LOG_DIR"

for DS in isic chestx crop_disease; do
  for L3 in 0.3 0; do
    echo ">>> [步驟 2 FSR 28×28, λ3=$L3, separate λ=1, value] $DS 5-shot  $(date '+%F %T')"
    $PYTHON train_episodic.py --dataset "$DS" --n_shot 5 --class_names full \
        --local_score --local_k_frac 0.1 --local_gamma 1.0 --patch_mode value \
        --local_loss separate --local_lambda 1.0 \
        --use_feature_sr --sr_scale 2 --sr_refiner_layers 0 --lambda3 "$L3" \
        --save_dir ./results_episodic \
        2>&1 | tee -a "$LOG_DIR/${DS}_5shot_fsr_l3${L3}.log"
  done
done
echo "全部完成 $(date '+%F %T')"
