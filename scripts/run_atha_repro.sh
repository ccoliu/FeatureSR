#!/bin/bash
# run_atha_repro.sh
# 用 ATHA 官方程式碼（github.com/shuaiyi308/ATHA，ICML 2026）重現其 Table 1，作為可驗證的 SOTA 對照與實驗框架：
#   baseline（-method none，CoOp+LoRA）  5-shot 目標：Crop 96.21 / EuroSAT 92.52 / ISIC 51.10 / ChestX 24.13（平均 65.99）
#                                         1-shot 目標：Crop 85.32 / EuroSAT 81.41 / ISIC 35.23 / ChestX 21.73（平均 55.92）
#   ATHA（-method ours）                  5-shot 目標：97.62 / 93.41 / 56.42 / 26.67（平均 68.53）
#                                         1-shot 目標：87.99 / 82.56 / 38.86 / 24.00（平均 58.35）
#
# 注意：
#   - README 的指令沒有 -method ours，照抄只會跑到 baseline；ATHA 只在 -method ours 時啟用（clip/model.py）。
#   - add_ratio / minus_ratio 只有 ISIC 的範例（0.3 / 0.2），其他資料集沿用同值，屬於推測。
#   - 官方程式碼需先套用 scripts/atha_fix_return_similarity.patch（呼叫 block 時多傳了未定義的參數）。
#   - 每個 episode 100 步（total_epoch=100、batch_size=9999 ⇒ 全 support 一批）；1-shot 800、5-shot 400 個 episode。
#
# 前置：
#   git clone https://github.com/shuaiyi308/ATHA.git ~/ATHA && cd ~/ATHA && git apply ~/FeatureSR/scripts/atha_fix_return_similarity.patch
#   cd ~/FeatureSR && python scripts/make_atha_json.py --out_dir ~/atha_data
#   pip install h5py
set -e
set -o pipefail

PYTHON=${PYTHON:-python}
ATHA_DIR=${ATHA_DIR:-$HOME/ATHA}
DATA_DIR=${DATA_DIR:-$HOME/atha_data}
LOG_DIR=${LOG_DIR:-$HOME/FeatureSR/logs/atha_repro}
mkdir -p "$LOG_DIR"
cd "$ATHA_DIR"

COMMON="-r 16 -alpha 8 -lora_lr 2e-4 -coop_lr 2e-3 -base_lr 0.001 -data_path $DATA_DIR"

for SHOT in 5 1; do
  for DS in ISIC EuroSAT CropDiseases ChestX; do
    echo ">>> [baseline] $DS ${SHOT}-shot  $(date '+%F %T')"
    $PYTHON coop_lora_trainer.py $COMMON -dataset "$DS" -n_shot "$SHOT" -method none \
        2>&1 | grep --line-buffered -v -i warn | tee "$LOG_DIR/${DS}_${SHOT}shot_baseline.log"
    echo ">>> [ATHA] $DS ${SHOT}-shot  $(date '+%F %T')"
    $PYTHON coop_lora_trainer.py $COMMON -dataset "$DS" -n_shot "$SHOT" -method ours -add_ratio 0.3 -minus_ratio 0.2 \
        2>&1 | grep --line-buffered -v -i warn | tee "$LOG_DIR/${DS}_${SHOT}shot_atha.log"
  done
done
echo "全部完成 $(date '+%F %T')"
grep -h "Test Acc" "$LOG_DIR"/*.log
