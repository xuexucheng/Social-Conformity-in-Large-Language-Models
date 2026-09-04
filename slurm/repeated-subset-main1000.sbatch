#!/bin/bash
#SBATCH --job-name=repeat30
#SBATCH --partition=cpudebug
#SBATCH --qos=cpudebug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=logs/repeat30_%j.out
#SBATCH --error=logs/repeat30_%j.err

set -euo pipefail

cd /gpfs/work/int/xuchengxue23/social_conformity_revision

POOL="agent_conformity_project/analysis/recovered_pool1000"
OUT="agent_conformity_project/analysis/repeated_subset_main1000"
PY="/gpfs/work/int/xuchengxue23/envs/self_history_vllm/bin/python"

echo "START=$(date)"
echo "HOST=$(hostname)"
free -h

rm -rf "$OUT"

$PY agent_conformity_project/analysis/repeated_subset_analysis.py \
  --manifest "$POOL/repeated30_manifest.json" \
  --output-dir "$OUT" \
  --sample-size 500 \
  --repetitions 30 \
  --seed-start 1000

echo "END=$(date)"
ls -lh "$OUT"
