#!/bin/bash
#SBATCH --job-name=p3-terciles
#SBATCH --partition=cpudebug
#SBATCH --qos=cpudebug
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:20:00
#SBATCH --output=/gpfs/work/int/xuchengxue23/social_conformity_revision/slurm/logs/p3_terciles_%j.out
#SBATCH --error=/gpfs/work/int/xuchengxue23/social_conformity_revision/slurm/logs/p3_terciles_%j.err

set -euo pipefail

PROJECT="/gpfs/work/int/xuchengxue23/social_conformity_revision/agent_conformity_project"

module purge
module load anaconda3/2022.10-gcc-8.5.0-4dp3trd

PY="$(command -v python)"

echo "===== RUNTIME ====="
echo "host=$(hostname)"
echo "python=$PY"
"$PY" --version
"$PY" -c 'print("PYTHON_RUNTIME_OK")'

cd "$PROJECT"
export PYTHONPATH="$PROJECT"

echo "===== START P3 ANALYSIS ====="
"$PY" analysis/analyze_forced_option_terciles.py
echo "===== END P3 ANALYSIS ====="
