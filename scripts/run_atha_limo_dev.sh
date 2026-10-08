#!/bin/bash
# run_atha_limo_dev.sh
# LIMO 移植到 ATHA 強 baseline（coop_lora_trainer_local.py --limo）的開發集：ISIC 5-shot、有增強、seed 2、100 個 episode。
# 我們的框架上 LIMO 在 ISIC 5-shot +3.32 ± 0.85；ATHA 的 CE 在 logit scale 4，LIMO 官方在 100，
# 因此事先固定兩個候選，於開發集選定後再到確認集（seed 1，400 個 episode）只跑選定的那一個：
#   s100：LIMO 損失用 scale 100（官方）    s0：LIMO 損失沿用 ATHA 的 scale 4
# 每步無標註數 = 每步 support 數（125，1:1，與我們框架上有效的比例相同）；權重為官方預設 1 / 0.1 / 10。
# 對照：results_atha/ISIC_5shot_base_seed2_n100.jsonl（同一批 episode，訓練與 baseline 相同）。
# 會先等 run_limo_5shot.sh 跑完（兩者同時跑顯存不足）。
set -e
set -o pipefail
FSR=$HOME/FeatureSR
PYTHON=${PYTHON:-$FSR/.venv/bin/python}
WAIT_LOG=$FSR/logs/limo_5shot_driver.log
mkdir -p "$FSR/logs/atha_limo"
if [ -f "$WAIT_LOG" ] && ! grep -q "LIMO_ALL_DONE" "$WAIT_LOG"; then
  echo "等待 run_limo_5shot.sh 跑完  $(date '+%F %T')"
  until grep -q "LIMO_ALL_DONE\|Traceback" "$WAIT_LOG"; do sleep 60; done
fi
cd "$HOME/ATHA"
for SC in 100 0; do
  J="$FSR/results_atha/ISIC_5shot_limo_s${SC}_seed2_n100.jsonl"
  if [ ! -f "$J" ]; then
    echo ">>> [ATHA LIMO 開發集] ISIC 5-shot limo_scale=$SC  $(date '+%F %T')"
    $PYTHON -u coop_lora_trainer_local.py -r 16 -alpha 8 -lora_lr 2e-4 -coop_lr 2e-3 -base_lr 0.001 -data_path $HOME/atha_data \
        -dataset ISIC -n_shot 5 -method none -aug --paired -seed 2 --n_episodes 100 --limo --limo_scale "$SC" \
        --out_jsonl "$J.partial" 2>&1 | grep --line-buffered -v -i warn | tee "$FSR/logs/atha_limo/ISIC_5shot_limo_s${SC}_seed2.log"
    mv "$J.partial" "$J"
  fi
  $PYTHON "$FSR/scripts/paired_diff.py" "$J" "$FSR/results_atha/ISIC_5shot_base_seed2_n100.jsonl" | sed "s/^/[ATHA_LIMO_DEV s$SC] /"
done
echo "ATHA_LIMO_DEV_DONE $(date '+%F %T')"
