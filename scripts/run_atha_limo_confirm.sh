#!/bin/bash
# run_atha_limo_confirm.sh
# LIMO（transductive 微調）在 ATHA 強 baseline 上的 5-shot 確認集：seed 1、400 個 episode（同 ATHA 5-shot 的 episode 數）。
# 設定鎖定為開發集選定的 limo_scale 0（LIMO 損失沿用 ATHA 的 scale 4；開發集 ISIC +0.65 ± 1.21、EuroSAT +3.29 ± 0.63）。
# 順序：EuroSAT → CropDiseases（BSCD 標準版，先補 baseline）→ ChestX。ISIC 開發集無增益，依流程不跑確認集。
# 對照：同一批 episode 的 baseline（EuroSAT / ChestX 為先前存特徵時的 baseline，訓練與 baseline 完全相同）。
# 每個 LIMO 約 7.8 小時（約 70 秒 / episode）。中斷後重跑會跳過已完成的部分（ATHA 程式不支援續跑，未完成者從頭開始）。
set -e
set -o pipefail
FSR=$HOME/FeatureSR
PYTHON=${PYTHON:-$FSR/.venv/bin/python}
mkdir -p "$FSR/logs/atha_limo"
run() {   # run <名稱> <ATHA 資料集名稱> <資料根目錄> <base|limo_s0>
  local NAME=$1 DS=$2 ROOT=$3 M=$4
  local J="$FSR/results_atha/${NAME}_5shot_${M}_seed1_n400.jsonl"
  [ -f "$J" ] && return 0
  local EXTRA=""; [ "$M" = limo_s0 ] && EXTRA="--limo --limo_scale 0"
  echo ">>> [ATHA LIMO 確認集] $NAME 5-shot $M  $(date '+%F %T')"
  cd "$HOME/ATHA"
  $PYTHON -u coop_lora_trainer_local.py -r 16 -alpha 8 -lora_lr 2e-4 -coop_lr 2e-3 -base_lr 0.001 -data_path "$ROOT" \
      -dataset "$DS" -n_shot 5 -method none -aug --paired -seed 1 --n_episodes 400 $EXTRA \
      --out_jsonl "$J.partial" 2>&1 | grep --line-buffered -v -i warn | tee "$FSR/logs/atha_limo/${NAME}_5shot_${M}_seed1.log"
  mv "$J.partial" "$J"
  cd "$FSR"
}
for SPEC in "EuroSAT EuroSAT $HOME/atha_data" "CropDiseases_bscd CropDiseases $HOME/atha_data_bscd" "ChestX ChestX $HOME/atha_data"; do
  set -- $SPEC
  NAME=$1; DS=$2; ROOT=$3
  run "$NAME" "$DS" "$ROOT" base
  run "$NAME" "$DS" "$ROOT" limo_s0
  $PYTHON "$FSR/scripts/paired_diff.py" "$FSR/results_atha/${NAME}_5shot_limo_s0_seed1_n400.jsonl" \
      "$FSR/results_atha/${NAME}_5shot_base_seed1_n400.jsonl" | sed "s/^/[LIMO_CONFIRM $NAME] /"
done
echo "LIMO_CONFIRM_ALL_DONE $(date '+%F %T')"
