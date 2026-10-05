#!/bin/bash
# run_transclip_crop_variants.sh
# 驗證 CropDiseases 1-shot 的 TransCLIP 增益（Augmented 版：+5.21 ± 0.34）是否被資料池中的離線增強副本放大。
# 用同一個鎖定設定 model2:0.002、seed 1、800 個 episode，依序跑：
#   bscd ：BSCD-FSL 原始 benchmark 的 CropDiseases（Kaggle saroz014/plant-disease 的 dataset/train，43,456 張）
#          清單：scripts/make_crop_bscd_json.py → ~/atha_data_bscd
#   dedup：從 Augmented 版依檔名濾掉增強副本（39,059 張）
#          清單：scripts/make_crop_dedup_json.py → ~/atha_data_dedup
# 若 GPU 上 run_transclip_5shot_all.sh 還沒跑完，會先等它（以 log 的 TRANSCLIP_5SHOT_ALL_DONE 為準）。
# 中斷後重跑：已完成的版本會跳過（ATHA 程式本身不支援續跑，未完成的版本從頭開始）。
set -e
set -o pipefail
FSR=$HOME/FeatureSR
PYTHON=${PYTHON:-$FSR/.venv/bin/python}
VARIANTS=${VARIANTS:-"bscd dedup"}
WAIT_LOG=$FSR/logs/transclip_5shot_all_driver.log
mkdir -p "$FSR/logs/atha_dump"

if [ -f "$WAIT_LOG" ] && ! grep -q "TRANSCLIP_5SHOT_ALL_DONE" "$WAIT_LOG"; then
  echo "等待 run_transclip_5shot_all.sh 跑完  $(date '+%F %T')"
  until grep -q "TRANSCLIP_5SHOT_ALL_DONE\|Traceback" "$WAIT_LOG"; do sleep 60; done
fi

for V in $VARIANTS; do
  ROOT=$HOME/atha_data_$V
  [ -f "$ROOT/CropDiseases/novel.json" ] || $PYTHON "$FSR/scripts/make_crop_${V}_json.py"
  DUMP=$HOME/atha_dumps/CropDiseases_${V}_1shot_seed1
  J=$FSR/results_atha/CropDiseases_${V}_1shot_base_seed1_n800.jsonl
  if [ ! -f "$J" ]; then
    echo ">>> [ATHA dump] CropDiseases（$V）1-shot seed=1 n=800  $(date '+%F %T')"
    cd "$HOME/ATHA"
    $PYTHON -u coop_lora_trainer_local.py -r 16 -alpha 8 -lora_lr 2e-4 -coop_lr 2e-3 -base_lr 0.001 \
        -data_path "$ROOT" -dataset CropDiseases -n_shot 1 -method none -aug --paired -seed 1 --n_episodes 800 \
        --out_jsonl "$J.partial" --dump_dir "$DUMP" \
        2>&1 | grep --line-buffered -v -i warn | tee "$FSR/logs/atha_dump/CropDiseases_${V}_1shot_seed1.log"
    mv "$J.partial" "$J"
    cd "$FSR"
  fi
  $PYTHON "$FSR/scripts/transclip_dump.py" "$DUMP" --only model2:0.002 2>&1 | grep -v -i warn \
      | tee "$FSR/results_atha/transclip_CropDiseases_${V}_1shot_seed1.txt"
  echo "CROP_VARIANT_DONE $V"
done
echo "CROP_VARIANTS_ALL_DONE $(date '+%F %T')"
