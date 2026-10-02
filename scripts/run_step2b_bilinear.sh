#!/bin/bash
# run_step2b_bilinear.sh
# 步驟 2b 診斷：步驟 2 的 learned 上採樣器（bilinear + 卷積 residual + 位置編碼 + refiner ln_post，約 592 萬參數，
# 每個 episode 隨機初始化）在 ISIC 5-shot 比 14×14 低 1.4 分。這裡換成純 bilinear（0 參數），λ3=0，
# 判斷是「上採樣器的參數破壞訊號」還是「28×28 上做 top-k 本身不利」。
# 對照：14×14（步驟 1）與 learned（步驟 2），皆同一批 episode。
set -e
set -o pipefail
cd "$(dirname "$0")/.."

PYTHON=${PYTHON:-.venv/bin/python}
LOG_DIR="logs/step2b_bilinear"
mkdir -p "$LOG_DIR"

for DS in isic; do
  echo ">>> [步驟 2b 純 bilinear 28×28, separate λ=1, value] $DS 5-shot  $(date '+%F %T')"
  $PYTHON train_episodic.py --dataset "$DS" --n_shot 5 --class_names full \
      --local_score --local_k_frac 0.1 --local_gamma 1.0 --patch_mode value \
      --local_loss separate --local_lambda 1.0 \
      --use_feature_sr --sr_scale 2 --sr_upsampler bilinear --lambda3 0 \
      --save_dir ./results_episodic \
      2>&1 | tee -a "$LOG_DIR/${DS}_5shot_bilinear.log"
done
echo "全部完成 $(date '+%F %T')"
