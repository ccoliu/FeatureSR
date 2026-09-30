#!/bin/bash
# smoke_step1.sh
# 步驟 1（patch 層級分數）程式碼的煙霧測試，約 6 分鐘。結果寫到 /tmp，不影響 results_episodic/。
#   1. value 模式下 CLS 與 plain 完全相同
#   2. 不開 --local_score 時，前 2 個 episode 與正式 cnfull baseline 逐 episode 相同（迴歸檢查）
#   3. local plain / local value / local + learn_gamma / local + FSR 各跑 1 個 episode，確認能跑、有記錄診斷欄位
set -e
set -o pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate
OUT=/tmp/smoke_step1
rm -rf "$OUT"; mkdir -p "$OUT"

echo "=== 1. value 模式 CLS 檢查 ==="
python -c "
import torch
from models.clip_wrapper import CLIPWrapper
from models.clip_lora import CLIPLoRA
w = CLIPWrapper(device='cuda')
m = CLIPLoRA(w, r=2, alpha=1, dropout=0.25, encoder='both', qkv_mode='separate', use_out_proj=False); m.eval()
x = torch.randn(2, 3, 224, 224, device='cuda')
with torch.no_grad():
    c1, p1 = m.encode_image_with_patches(x)
    c2, p2 = m.encode_image_with_patches(x, patch_mode='value')
print(f'CLS 最大差 {(c1 - c2).abs().max().item():.2e}（應為 0）| patch {tuple(p2.shape)} | plain·value patch 平均 cos {(p1 * p2).sum(-1).mean().item():.3f}')
"

COMMON="--dataset isic --n_shot 5 --save_dir $OUT --log_every 1"
run() { python train_episodic.py $COMMON "$@" 2>&1 | grep -E "tag=|Final|診斷|Error|error" ; }

echo "=== 2. 迴歸檢查（不開 local，2 個 episode）==="
run --n_episodes 2
python scripts/paired_diff.py $OUT/isic_5shot_nofsr_nocc_loraboth_r2_steps500_cnfull.jsonl \
    results_episodic/isic_5shot_nofsr_nocc_loraboth_r2_steps500_cnfull.jsonl
echo "（上面 B − A 應為 +0.00 ± 0.00）"

echo "=== 3. 各變體 1 個 episode ==="
run --n_episodes 1 --local_score
run --n_episodes 1 --local_score --patch_mode value
run --n_episodes 1 --local_score --learn_gamma
run --n_episodes 1 --local_score --use_feature_sr --sr_refiner_layers 0 --lambda3 0.3
for f in $OUT/*loc*.jsonl; do
  python -c "
import json, sys
r = json.loads(open('$f').readline())
print(f\"{'$f'.split('/')[-1]:75s} acc {r['acc']:.2f} | cls {r['acc_cls']:.2f} | local {r['acc_local']:.2f} | γ {r['gamma']:.3f}\")
"
done
echo "煙霧測試完成"
