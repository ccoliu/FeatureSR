#!/bin/bash
# run_transclip_1shot_all.sh
# TransCLIP（transductive）推廣到其餘資料集的 1-shot。設定已在 ISIC 1-shot 鎖定：
#   開發集 seed 2（100 個 episode）從 8 個候選選出 model2:0.002（+1.07 ± 0.75），
#   確認集 seed 1（800 個 episode）+1.05 ± 0.25。
# 這裡每個資料集只評估這一個設定，seed 1、800 個 episode（同 ATHA 1-shot 的 episode 數）。
set -e
set -o pipefail
cd "$(dirname "$0")/.."
for DS in EuroSAT CropDiseases ChestX; do
  SHOT=1 AUG=1 SEED=1 N=800 DS=$DS bash scripts/run_atha_dump.sh
  .venv/bin/python scripts/transclip_dump.py ~/atha_dumps/${DS}_1shot_seed1 --only model2:0.002 2>&1 | grep -v -i warn \
      | tee results_atha/transclip_${DS}_1shot_seed1.txt
done
echo "全部完成 $(date '+%F %T')"
