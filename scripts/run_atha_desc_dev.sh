#!/bin/bash
# run_atha_desc_dev.sh
# 方向 A：LLM 類別描述（文字端，inductive）在 ATHA 強 baseline 上的開發集。
# 描述：configs/class_descriptions.json（每類一句，只根據類別名稱撰寫，不看影像）。
# 事先固定的候選（--desc_mode）：
#   cat ：CoOp prompt 的類別名稱換成「名稱, 描述」
#   only：只用描述（不含類別名稱）
# 開發集：ISIC、seed 2、100 個 episode，1-shot 與 5-shot；對照為同一批 episode 的既有 baseline
#   （results_atha/ISIC_{1,5}shot_base_seed2_n100.jsonl；冒煙測試已確認不加描述時逐 episode 相同）。
# 選擇規則：各 shot 取配對差平均較大的候選；CI 不含 0 且為正，該 shot 才進確認集（seed 1，四個資料集，設定鎖定）。
set -e
set -o pipefail
FSR=$HOME/FeatureSR
PYTHON=${PYTHON:-$FSR/.venv/bin/python}
DESC=$FSR/configs/class_descriptions.json
mkdir -p "$FSR/logs/atha_desc"
for K in 1 5; do
  for M in cat only; do
    J="$FSR/results_atha/ISIC_${K}shot_desc${M}_seed2_n100.jsonl"
    if [ ! -f "$J" ]; then
      echo ">>> [ATHA 描述開發集] ISIC ${K}-shot desc=$M  $(date '+%F %T')"
      cd "$HOME/ATHA"
      $PYTHON -u coop_lora_trainer_local.py -r 16 -alpha 8 -lora_lr 2e-4 -coop_lr 2e-3 -base_lr 0.001 \
          -data_path "$HOME/atha_data" -dataset ISIC -n_shot $K -method none -aug --paired -seed 2 --n_episodes 100 \
          --desc_file "$DESC" --desc_mode $M --out_jsonl "$J.partial" \
          2>&1 | grep --line-buffered -v -i warn | tee "$FSR/logs/atha_desc/ISIC_${K}shot_desc${M}_seed2.log"
      mv "$J.partial" "$J"
      cd "$FSR"
    fi
    $PYTHON "$FSR/scripts/paired_diff.py" "$J" "$FSR/results_atha/ISIC_${K}shot_base_seed2_n100.jsonl" \
        | sed "s/^/[DESC_DEV ISIC ${K}shot $M] /"
  done
done
echo "DESC_DEV_ALL_DONE $(date '+%F %T')"
