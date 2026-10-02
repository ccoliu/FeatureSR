#!/bin/bash
# run_atha_local.sh
# 強 baseline 驗證：在 ATHA 官方程式碼的 CLIP-LoRA baseline（有增強，ISIC 5-shot 原本 54.98）上加入局部分數。
# 使用 ~/ATHA/coop_lora_trainer_local.py（由 coop_lora_trainer_aug.py 修改，差異見 scripts/atha_local.patch）。
# --paired：類別抽樣用獨立 generator、每個 episode 重設種子，baseline 與 --local 跑同一批 episode，可直接配對：
#   python scripts/paired_diff.py results_atha/ISIC_5shot_local.jsonl results_atha/ISIC_5shot_base.jsonl
# 每組約 3.8 小時。python -u 避免輸出被暫存（否則 log 看起來好幾分鐘沒進度）。
set -e
set -o pipefail
FSR=$HOME/FeatureSR
PYTHON=${PYTHON:-$FSR/.venv/bin/python}
OUT=$FSR/results_atha
LOG=$FSR/logs/atha_local
mkdir -p "$OUT" "$LOG"
cd "$HOME/ATHA"

COMMON="-r 16 -alpha 8 -lora_lr 2e-4 -coop_lr 2e-3 -base_lr 0.001 -data_path $HOME/atha_data -method none -aug --paired"
for DS in ISIC; do
  # local_l0：訓練時不用 local loss（λ=0），只在推論時融合，用來區分「local 訓練干擾 CLS」與「local 分數本身沒有補充資訊」
  for M in base local local_l0; do
    EXTRA=""
    [ "$M" = local ] && EXTRA="--local --local_lambda 1.0 --local_k_frac 0.1 --local_gamma 1.0"
    [ "$M" = local_l0 ] && EXTRA="--local --local_lambda 0 --local_k_frac 0.1 --local_gamma 1.0"
    J="$OUT/${DS}_5shot_${M}.jsonl"
    [ -f "$J" ] && { echo "已存在 $J，跳過（ATHA 程式碼不支援續跑，要重跑請先刪除）"; continue; }
    echo ">>> [ATHA 強 baseline, $M] $DS 5-shot  $(date '+%F %T')"
    $PYTHON -u coop_lora_trainer_local.py $COMMON -dataset "$DS" -n_shot 5 $EXTRA --out_jsonl "$J.partial" \
        2>&1 | grep --line-buffered -v -i warn | tee "$LOG/${DS}_5shot_${M}.log"
    mv "$J.partial" "$J"
  done
done
echo "全部完成 $(date '+%F %T')"
