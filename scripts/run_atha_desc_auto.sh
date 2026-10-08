#!/bin/bash
# run_atha_desc_auto.sh
# 方向 A（LLM 類別描述）：等 run_atha_desc_dev.sh（ISIC 開發集）跑完，依事先固定的規則自動接續。
#
# 選擇規則（各 shot 分開）：
#   mode_K = ISIC 開發集（seed 2，n=100）上配對差平均較大的候選（cat / only）
#   pass_K = mode_K 的配對差 95% CI 下界 > 0
# 情況一：至少一個 shot 通過 → 確認集（seed 1；1-shot n=800、5-shot n=400；設定鎖定為 mode_K）
#         順序：ISIC → ChestX → EuroSAT → CropDiseases（BSCD 標準版），同一資料集內先 1-shot 再 5-shot。
#         對照為同一批 episode 的既有 baseline（results_atha/{名稱}_{K}shot_base_seed1_n{N}.jsonl）。
# 情況二：兩個 shot 都沒通過 → 第二批開發集（seed 2，n=100，mode_K 沿用 ISIC 的選擇）：
#         EuroSAT 5-shot → EuroSAT 1-shot → ChestX 1-shot → ChestX 5-shot（缺的 baseline 一併補跑）。
#         比照 LIMO（ISIC 無增益、EuroSAT 有增益）的流程；是否進確認集留待人工決定。
# 中斷後重跑：已完成的結果檔會跳過（ATHA 程式不支援續跑，未完成者從頭開始）。
set -e
set -o pipefail
FSR=$HOME/FeatureSR
PYTHON=${PYTHON:-$FSR/.venv/bin/python}
DESC=$FSR/configs/class_descriptions.json
DEV_LOG=$FSR/logs/atha_desc_dev_driver.log
mkdir -p "$FSR/logs/atha_desc"

echo "等待 ISIC 開發集跑完  $(date '+%F %T')"
until grep -q "DESC_DEV_ALL_DONE" "$DEV_LOG"; do
  if grep -qE "Traceback|Killed" "$DEV_LOG"; then echo "DESC_AUTO_ABORT 開發集出錯"; exit 1; fi
  sleep 60
done

# run_atha <名稱> <ATHA 資料集> <資料根目錄> <shot> <seed> <n> <base|cat|only>
run_atha() {
  local NAME=$1 DS=$2 ROOT=$3 K=$4 S=$5 N=$6 M=$7
  local TAG=base EXTRA=""
  [ "$M" != base ] && TAG=desc$M && EXTRA="--desc_file $DESC --desc_mode $M"
  local J="$FSR/results_atha/${NAME}_${K}shot_${TAG}_seed${S}_n${N}.jsonl"
  [ -f "$J" ] && return 0
  echo ">>> [ATHA 描述] $NAME ${K}-shot $TAG seed=$S n=$N  $(date '+%F %T')"
  cd "$HOME/ATHA"
  $PYTHON -u coop_lora_trainer_local.py -r 16 -alpha 8 -lora_lr 2e-4 -coop_lr 2e-3 -base_lr 0.001 \
      -data_path "$ROOT" -dataset "$DS" -n_shot $K -method none -aug --paired -seed $S --n_episodes $N $EXTRA \
      --out_jsonl "$J.partial" 2>&1 | grep --line-buffered -v -i warn \
      | tee "$FSR/logs/atha_desc/${NAME}_${K}shot_${TAG}_seed${S}.log"
  mv "$J.partial" "$J"
  cd "$FSR"
}

# 依 ISIC 開發集選定各 shot 的設定
SEL=$($PYTHON - <<EOF
import json, numpy as np
R = "$FSR/results_atha/ISIC_%dshot_%s_seed2_n100.jsonl"
load = lambda p: {r["episode"]: r["acc"] for r in map(json.loads, open(p))}
for k in (1, 5):
    base = load(R % (k, "base"))
    stats = {}
    for m in ("cat", "only"):
        d = load(R % (k, "desc" + m))
        x = np.array([d[e] - base[e] for e in d])
        stats[m] = (x.mean(), 1.96 * x.std(ddof=1) / np.sqrt(len(x)))
    m = max(stats, key=lambda m: stats[m][0])
    mu, ci = stats[m]
    print(k, m, int(mu - ci > 0), "%+.2f±%.2f" % (mu, ci))
EOF
)
echo "$SEL" | sed "s/^/[DESC_SELECT] shot 設定 通過 配對差：/"

PASS=""; declare -A MODE
while read -r K M P _; do
  MODE[$K]=$M
  [ "$P" = 1 ] && PASS="$PASS $K"
done <<< "$SEL"

if [ -n "$PASS" ]; then
  echo "[DESC_PLAN] 確認集：shot$PASS"
  for SPEC in "ISIC ISIC $HOME/atha_data" "ChestX ChestX $HOME/atha_data" "EuroSAT EuroSAT $HOME/atha_data" \
              "CropDiseases_bscd CropDiseases $HOME/atha_data_bscd"; do
    set -- $SPEC; NAME=$1; DS=$2; ROOT=$3
    for K in $PASS; do
      N=$([ $K = 1 ] && echo 800 || echo 400)
      run_atha "$NAME" "$DS" "$ROOT" $K 1 $N "${MODE[$K]}"
      $PYTHON "$FSR/scripts/paired_diff.py" "$FSR/results_atha/${NAME}_${K}shot_desc${MODE[$K]}_seed1_n${N}.jsonl" \
          "$FSR/results_atha/${NAME}_${K}shot_base_seed1_n${N}.jsonl" | sed "s/^/[DESC_CONFIRM $NAME ${K}shot] /"
    done
  done
else
  echo "[DESC_PLAN] ISIC 兩個 shot 都未通過 → 第二批開發集（EuroSAT、ChestX）"
  for SPEC in "EuroSAT 5" "EuroSAT 1" "ChestX 1" "ChestX 5"; do
    set -- $SPEC; NAME=$1; K=$2
    run_atha "$NAME" "$NAME" "$HOME/atha_data" $K 2 100 base
    run_atha "$NAME" "$NAME" "$HOME/atha_data" $K 2 100 "${MODE[$K]}"
    $PYTHON "$FSR/scripts/paired_diff.py" "$FSR/results_atha/${NAME}_${K}shot_desc${MODE[$K]}_seed2_n100.jsonl" \
        "$FSR/results_atha/${NAME}_${K}shot_base_seed2_n100.jsonl" | sed "s/^/[DESC_DEV2 $NAME ${K}shot] /"
  done
fi
echo "DESC_AUTO_ALL_DONE $(date '+%F %T')"
