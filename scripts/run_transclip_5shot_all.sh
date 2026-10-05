#!/bin/bash
# run_transclip_5shot_all.sh
# TransCLIP 推廣到 5-shot（seed 1，每個資料集 400 個 episode，同 ATHA 5-shot 的 episode 數）。
# 設定沿用 ISIC 5-shot 開發集（seed 2，n=100）依規則選出的 model4:0.002（該設定在開發集上 +0.29 ± 0.47，不顯著）；
# 1-shot 的增益依資料集差異很大（ISIC 最小），因此仍推廣到各資料集確認。
# 順序依預期增益：EuroSAT → ISIC → CropDiseases → ChestX。每個約 3.9 小時。
set -e
set -o pipefail
cd "$(dirname "$0")/.."
for DS in EuroSAT ISIC CropDiseases ChestX; do
  SHOT=5 AUG=1 SEED=1 N=400 DS=$DS bash scripts/run_atha_dump.sh
  .venv/bin/python scripts/transclip_dump.py ~/atha_dumps/${DS}_5shot_seed1 --only model4:0.002 2>&1 | grep -v -i warn \
      | tee results_atha/transclip_${DS}_5shot_seed1.txt
  echo "TRANSCLIP_5SHOT_DONE $DS"
done
echo "TRANSCLIP_5SHOT_ALL_DONE $(date '+%F %T')"
