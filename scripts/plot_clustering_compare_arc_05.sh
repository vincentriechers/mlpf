#!/bin/bash
#SBATCH --job-name=mlpf_plot_clust_cmp
#SBATCH --output=/gpfs/scratch/ehpc399/vincent/logs/%x-%j.out
#SBATCH --error=/gpfs/scratch/ehpc399/vincent/logs/%x-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --qos=acc_ehpc
#SBATCH --account=ehpc399

set -euo pipefail

ml miniforge/24.3.0-0
CONDA_BASE="$(dirname "$(dirname "$(which conda)")")"
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate /gpfs/scratch/ehpc399/vincent/envs/HitPF
PYTHON_BIN="/gpfs/scratch/ehpc399/vincent/envs/HitPF/bin/python"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Expected python not found: ${PYTHON_BIN}"
  exit 1
fi

cd /gpfs/scratch/ehpc399/vincent/code/mlpf
pwd
echo "Using python: ${PYTHON_BIN}"
"${PYTHON_BIN}" -c "import sys; print(sys.executable)"

ARC_HITPF="/gpfs/scratch/ehpc399/vincent/models/arc_clustering_1M_2303/showers_df_evaluation/eval_clustering_1M_arc_*.pkl0_0_None.pt"
O5_HITPF="/gpfs/scratch/ehpc399/vincent/models/05_clustering_1M_2303/showers_df_evaluation/eval_clustering_1M_05_*.pkl0_0_None.pt"
ARC_PANDORA="/gpfs/scratch/ehpc399/vincent/models/arc_clustering_1M_2303/showers_df_evaluation/eval_clustering_1M_arc_*_pandora.pt"
O5_PANDORA="/gpfs/scratch/ehpc399/vincent/models/05_clustering_1M_2303/showers_df_evaluation/eval_clustering_1M_05_*_pandora.pt"
OUTPUT_DIR="/gpfs/scratch/ehpc399/vincent/models/clustering_compare_1M_arc_05"

mkdir -p "${OUTPUT_DIR}"

CMD=(
  "${PYTHON_BIN}" -m src.evaluation.clustering
  --mlpf "${ARC_HITPF}" "${O5_HITPF}"
  --labels "ARC HitPF" "05 HitPF"
  --pandora "${ARC_PANDORA}" "${O5_PANDORA}"
  --pandora-labels "ARC Pandora" "05 Pandora"
  --output-dir "${OUTPUT_DIR}"
)

printf 'Running command:\n%s\n' "${CMD[*]}"
"${CMD[@]}"
