#!/bin/bash
# run_transclip_crop_dedup.sh
# CropDiseases 1-shot 的 TransCLIP 增益（+5.21 ± 0.34）可能被資料池中的離線增強副本放大：
# Kaggle「Augmented」版裡同一張原圖的翻轉 / 旋轉副本可能同時出現在一個 episode，TransCLIP 的 query kNN 圖會直接利用。
# 這裡改用 scripts/make_crop_dedup_json.py 產生的去重複清單（只留原圖，39,059 張、38 類），
# 以同一個鎖定設定 model2:0.002、seed 1、800 個 episode 重跑，比較增益是否縮小。
# （另寫一支腳本而不修改 run_atha_dump.sh：該檔正被 run_transclip_5shot_all.sh 逐一呼叫，執行中修改會出錯。）
set -e
set -o pipefail
FSR=$HOME/FeatureSR
PYTHON=${PYTHON:-$FSR/.venv/bin/python}
DUMP=$HOME/atha_dumps/CropDiseases_dedup_1shot_seed1
J=$FSR/results_atha/CropDiseases_dedup_1shot_base_seed1_n800.jsonl
mkdir -p "$FSR/logs/atha_dump"
[ -f "$HOME/atha_data_dedup/CropDiseases/novel.json" ] || $PYTHON "$FSR/scripts/make_crop_dedup_json.py"
if [ ! -f "$J" ]; then
  echo ">>> [ATHA dump] CropDiseases（去重複）1-shot seed=1 n=800  $(date '+%F %T')"
  cd "$HOME/ATHA"
  $PYTHON -u coop_lora_trainer_local.py -r 16 -alpha 8 -lora_lr 2e-4 -coop_lr 2e-3 -base_lr 0.001 \
      -data_path "$HOME/atha_data_dedup" -dataset CropDiseases -n_shot 1 -method none -aug --paired -seed 1 --n_episodes 800 \
      --out_jsonl "$J.partial" --dump_dir "$DUMP" \
      2>&1 | grep --line-buffered -v -i warn | tee "$FSR/logs/atha_dump/CropDiseases_dedup_1shot_seed1.log"
  mv "$J.partial" "$J"
  cd "$FSR"
fi
$PYTHON "$FSR/scripts/transclip_dump.py" "$DUMP" --only model2:0.002 2>&1 | grep -v -i warn \
    | tee "$FSR/results_atha/transclip_CropDiseases_dedup_1shot_seed1.txt"
echo "CROP_DEDUP_DONE $(date '+%F %T')"
