#!/bin/bash
# run_diag_local_training.sh
# 診斷局部分數在 ATHA 上失效的原因（10/2）：
#   1–2. ATHA 無增強（seed 2，100 個 episode，--paired）加 local 監督 λ=1：local 分支 logit scale = 4（同 CLS）與 100。
#        scale 4 時 local CE 無法飽和、會持續拉扯 CLS；scale 100 讓它能降到 0。
#        對照：results_atha/ISIC_5shot_noaug_base_seed2_n100.jsonl（同一批 episode）
#   3.   我們的框架 λ=0（訓練不用 local loss，只在推論融合），ISIC 5-shot 前 100 個 episode。
#        對照：步驟 1b 的 λ=1（..._sep1_pmvalue.jsonl）與 baseline（..._cnfull.jsonl）的同一批 episode。
#        回答：我們框架裡的融合增益，是局部資訊本來就有的互補性，還是 local 監督訓練出來的。
set -e
set -o pipefail
FSR=$HOME/FeatureSR
PYTHON=${PYTHON:-$FSR/.venv/bin/python}
mkdir -p "$FSR/results_atha" "$FSR/logs/diag_local_training"

cd "$HOME/ATHA"
for SC in 4 100; do
  J="$FSR/results_atha/ISIC_5shot_noaug_local_s${SC}_seed2_n100.jsonl"
  [ -f "$J" ] && { echo "已存在 $J，跳過"; continue; }
  echo ">>> [ATHA 無增強 local λ=1 scale=$SC] ISIC 5-shot seed 2  $(date '+%F %T')"
  $PYTHON -u coop_lora_trainer_local.py -r 16 -alpha 8 -lora_lr 2e-4 -coop_lr 2e-3 -base_lr 0.001 -data_path $HOME/atha_data \
      -dataset ISIC -n_shot 5 -method none --paired -seed 2 --n_episodes 100 \
      --local --local_lambda 1.0 --local_k_frac 0.1 --local_gamma 1.0 --local_scale "$SC" \
      --out_jsonl "$J.partial" 2>&1 | grep --line-buffered -v -i warn | tee "$FSR/logs/diag_local_training/atha_noaug_local_s${SC}.log"
  mv "$J.partial" "$J"
done

cd "$FSR"
echo ">>> [我們的框架 λ=0] ISIC 5-shot 100 episodes  $(date '+%F %T')"
$PYTHON train_episodic.py --dataset isic --n_shot 5 --n_episodes 100 --class_names full \
    --local_score --local_k_frac 0.1 --local_gamma 1.0 --patch_mode value \
    --local_loss separate --local_lambda 0 --save_dir ./results_episodic \
    2>&1 | tee -a logs/diag_local_training/ours_lambda0.log
echo "全部完成 $(date '+%F %T')"
