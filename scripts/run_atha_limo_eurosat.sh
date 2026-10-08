#!/bin/bash
# run_atha_limo_eurosat.sh
# LIMO 在 ATHA 強 baseline 上的第二個資料集：EuroSAT 5-shot、有增強、seed 2、100 個 episode。
# ISIC 5-shot 開發集上選定 limo_scale 0（LIMO 損失沿用 ATHA 的 scale 4；+0.65 ± 1.21，scale 100 為 −18.27）；
# 我們的框架上 EuroSAT 5-shot 的 LIMO 增益最大（+3.43），用來確認「強 baseline 上無增益」是否只是 ISIC 的特例。
# 先跑 baseline（--paired，同一批 episode），再跑 LIMO，最後配對比較。
set -e
set -o pipefail
FSR=$HOME/FeatureSR
PYTHON=${PYTHON:-$FSR/.venv/bin/python}
mkdir -p "$FSR/logs/atha_limo"
cd "$HOME/ATHA"
C="-r 16 -alpha 8 -lora_lr 2e-4 -coop_lr 2e-3 -base_lr 0.001 -data_path $HOME/atha_data -dataset EuroSAT -n_shot 5 -method none -aug --paired -seed 2 --n_episodes 100"
for M in base limo_s0; do
  J="$FSR/results_atha/EuroSAT_5shot_${M}_seed2_n100.jsonl"
  [ -f "$J" ] && continue
  EXTRA=""; [ "$M" = limo_s0 ] && EXTRA="--limo --limo_scale 0"
  echo ">>> [ATHA EuroSAT 5-shot] $M  $(date '+%F %T')"
  $PYTHON -u coop_lora_trainer_local.py $C $EXTRA --out_jsonl "$J.partial" 2>&1 | grep --line-buffered -v -i warn \
      | tee "$FSR/logs/atha_limo/EuroSAT_5shot_${M}_seed2.log"
  mv "$J.partial" "$J"
done
$PYTHON "$FSR/scripts/paired_diff.py" "$FSR/results_atha/EuroSAT_5shot_limo_s0_seed2_n100.jsonl" \
    "$FSR/results_atha/EuroSAT_5shot_base_seed2_n100.jsonl" | sed "s/^/[ATHA_LIMO_EUROSAT] /"
echo "ATHA_LIMO_EUROSAT_DONE $(date '+%F %T')"
