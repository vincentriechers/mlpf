#!/bin/bash
#SBATCH --job-name=mlpf_plot_full_cmp
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

ARC_HITPF="/gpfs/scratch/ehpc399/vincent/models/arc_properties_1M_2403/showers_df_evaluation/eval_full_1M_arc_*.pkl0_0_None.pt"
O5_HITPF="/gpfs/scratch/ehpc399/vincent/models/05_properties_1M_2403/showers_df_evaluation/eval_full_1M_05_*.pkl0_0_None.pt"
ARC_PANDORA="/gpfs/scratch/ehpc399/vincent/models/arc_properties_1M_2403/showers_df_evaluation/eval_full_1M_arc_*_pandora.pt"
O5_PANDORA="/gpfs/scratch/ehpc399/vincent/models/05_properties_1M_2403/showers_df_evaluation/eval_full_1M_05_*_pandora.pt"
OUTPUT_DIR="/gpfs/scratch/ehpc399/vincent/models/full_evaluation_compare_1M_arc_05"
TEXLIVE_CACHE_BASE="/gpfs/scratch/ehpc399/vincent/.texlive2020"

export TEXMFVAR="${TEXLIVE_CACHE_BASE}/texmf-var"
export TEXMFCONFIG="${TEXLIVE_CACHE_BASE}/texmf-config"
export TEXMFHOME="${TEXLIVE_CACHE_BASE}/texmf-home"
export MLPF_PLOT_USETEX=1

mkdir -p "${OUTPUT_DIR}"
mkdir -p "${TEXMFVAR}" "${TEXMFCONFIG}" "${TEXMFHOME}"

echo "Using TeX cache:"
echo "  TEXMFVAR=${TEXMFVAR}"
echo "  TEXMFCONFIG=${TEXMFCONFIG}"
echo "  TEXMFHOME=${TEXMFHOME}"

latex -interaction=nonstopmode \
  --halt-on-error \
  --output-directory "${TEXLIVE_CACHE_BASE}" \
  /home/cern/cern948842/latex_smoke/smoke.tex \
  >"${TEXLIVE_CACHE_BASE}/job_latex_smoke.log" 2>&1

CMD=(
  "${PYTHON_BIN}" -m src.evaluation.full_evaluation
  --arc-mlpf "${ARC_HITPF}"
  --o5-mlpf "${O5_HITPF}"
  --arc-pandora "${ARC_PANDORA}"
  --o5-pandora "${O5_PANDORA}"
  --datatype "hitpf_pandora_compare"
  --output-dir "${OUTPUT_DIR}"
)

printf 'Running command:\n%s\n' "${CMD[*]}"
"${CMD[@]}"
