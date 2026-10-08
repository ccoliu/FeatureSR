#!/bin/bash
# run_limo_5shot.sh
# LIMO（transductive 微調）移植到我們的 CLIP-LoRA 框架（train_episodic.py --limo），5-shot。
# TransCLIP（transductive 推論）在 5-shot 只有約 +0.2；LIMO 論文報告 4-shot 比 CLIP-LoRA 平均 +2.8%。
# 權重用 LIMO 官方預設（條件熵 1、文字 KL 0.1、邊際 KL 10），未經挑選。
# 與官方的差異：每步保留全部 support（與 baseline 的 CE 完全相同）＋ 24 個無標註 query（官方 batch 32、1:3）；
# 總步數維持 500（官方 500 × shots）。
# 對照組：步驟 0 的 baseline（..._steps500_cnfull.jsonl，同一批 episode）。
# 先各跑 200 個 episode（每組約 2.7 小時）；要擴大到 400 個時，把 N 改成 400 重跑同一腳本即可續跑。
set -e
set -o pipefail
cd "$(dirname "$0")/.."
PYTHON=${PYTHON:-.venv/bin/python}
N=${N:-200}
mkdir -p logs/limo
for DS in isic eurosat; do
  echo ">>> [LIMO 5-shot] $DS n=$N  $(date '+%F %T')"
  $PYTHON train_episodic.py --dataset "$DS" --n_shot 5 --n_episodes "$N" --class_names full --limo \
      --save_dir ./results_episodic 2>&1 | tee -a "logs/limo/${DS}_5shot_limo.log"
  $PYTHON scripts/paired_diff.py results_episodic/${DS}_5shot_nofsr_nocc_loraboth_r2_steps500_cnfull_limo24.jsonl \
      results_episodic/${DS}_5shot_nofsr_nocc_loraboth_r2_steps500_cnfull.jsonl | sed "s/^/[LIMO_RESULT $DS] /"
done
echo "LIMO_ALL_DONE $(date '+%F %T')"
