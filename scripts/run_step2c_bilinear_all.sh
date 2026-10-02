#!/bin/bash
# run_step2c_bilinear_all.sh
# 步驟 2c：
#   1. ISIC 5-shot：14×14 + 3×3 平滑後取 top-k。純 bilinear 28×28 比 14×14 高 +0.30 ± 0.29，
#      檢驗這是否只是「較平滑的池化」效果，而不是上採樣帶來的資訊。
#   2. ChestX、CropDiseases 5-shot：純 bilinear 28×28，確認步驟 2b 的結果能否推廣。
# 設定皆同步驟 1 最終設定（value patch、分開監督 λ=1、推論 γ=1）。中斷後重跑同一腳本會自動續跑。
set -e
set -o pipefail
cd "$(dirname "$0")/.."

PYTHON=${PYTHON:-.venv/bin/python}
LOG_DIR="logs/step2c"
mkdir -p "$LOG_DIR"
COMMON="--n_shot 5 --class_names full --local_score --local_k_frac 0.1 --local_gamma 1.0 --patch_mode value
        --local_loss separate --local_lambda 1.0 --save_dir ./results_episodic"

echo ">>> [步驟 2c 14×14 + 3×3 平滑] isic 5-shot  $(date '+%F %T')"
$PYTHON train_episodic.py --dataset isic $COMMON --patch_smooth 3 \
    2>&1 | tee -a "$LOG_DIR/isic_5shot_smooth3.log"

for DS in chestx crop_disease; do
  echo ">>> [步驟 2c 純 bilinear 28×28] $DS 5-shot  $(date '+%F %T')"
  $PYTHON train_episodic.py --dataset "$DS" $COMMON --use_feature_sr --sr_scale 2 --sr_upsampler bilinear --lambda3 0 \
      2>&1 | tee -a "$LOG_DIR/${DS}_5shot_bilinear.log"
done
echo "全部完成 $(date '+%F %T')"
