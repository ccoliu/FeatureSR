#!/bin/bash
# run_atha_dump.sh
# 在 ATHA baseline（有增強、ISIC 5-shot）上存下推論時的特徵，供 scripts/analyze_dump.py 離線比較各種推論時的訊號
# （patch 對文字、DN4 式 patch 對 support patch、support 原型、不同融合方式）。訓練與主指標與 baseline 完全相同（已驗證）。
#   開發集：SEED=2 N=100（在這裡選方法與參數）
#   確認集：SEED=1 N=400（與 results_atha/ISIC_5shot_base.jsonl 同一批 episode，只用鎖定的設定跑一次）
#   AUG=0：不加 -aug（無增強的 ATHA baseline，用來檢驗「增強是否吸收了局部分數的資訊」）
# 每個 episode 約 40 MB，100 個約 4 GB；存在 repo 之外的 ~/atha_dumps。
set -e
set -o pipefail
SEED=${SEED:-2}
N=${N:-100}
DS=${DS:-ISIC}
SHOT=${SHOT:-5}
AUG=${AUG:-1}
if [ "$AUG" = 1 ]; then AUGFLAG=-aug; TAG=""; else AUGFLAG=""; TAG="_noaug"; fi
FSR=$HOME/FeatureSR
PYTHON=${PYTHON:-$FSR/.venv/bin/python}
DUMP=$HOME/atha_dumps/${DS}_${SHOT}shot${TAG}_seed${SEED}
mkdir -p "$FSR/results_atha" "$FSR/logs/atha_dump"
cd "$HOME/ATHA"
J="$FSR/results_atha/${DS}_${SHOT}shot${TAG}_base_seed${SEED}_n${N}.jsonl"
[ -f "$J" ] && { echo "已存在 $J，跳過"; exit 0; }
echo ">>> [ATHA dump] $DS ${SHOT}-shot aug=$AUG seed=$SEED n=$N  $(date '+%F %T')"
$PYTHON -u coop_lora_trainer_local.py -r 16 -alpha 8 -lora_lr 2e-4 -coop_lr 2e-3 -base_lr 0.001 -data_path $HOME/atha_data \
    -dataset "$DS" -n_shot "$SHOT" -method none $AUGFLAG --paired -seed "$SEED" --n_episodes "$N" \
    --out_jsonl "$J.partial" --dump_dir "$DUMP" \
    2>&1 | grep --line-buffered -v -i warn | tee "$FSR/logs/atha_dump/${DS}_${SHOT}shot${TAG}_seed${SEED}.log"
mv "$J.partial" "$J"
echo "全部完成 $(date '+%F %T')"
