#!/usr/bin/env bash

set -euo pipefail

ml miniforge/24.3.0-0
CONDA_BASE="$(dirname "$(dirname "$(which conda)")")"
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate /gpfs/scratch/ehpc399/vincent/envs/HitPF

export TEXMFVAR="/gpfs/scratch/ehpc399/vincent/.texlive2020/texmf-var"
export TEXMFCONFIG="/gpfs/scratch/ehpc399/vincent/.texlive2020/texmf-config"
export TEXMFHOME="/gpfs/scratch/ehpc399/vincent/.texlive2020/texmf-home"
export MLPF_PLOT_USETEX=1

echo "Using TeX cache:"
echo "  TEXMFVAR="
echo "  TEXMFCONFIG="
echo "  TEXMFHOME="
latex -interaction=nonstopmode --halt-on-error --output-directory /gpfs/scratch/ehpc399/vincent/.texlive2020 /home/cern/cern948842/latex_smoke/smoke.tex >/gpfs/scratch/ehpc399/vincent/.texlive2020/job_latex_smoke.log 2>&1

# Full-evaluation outputs for the current 1M properties regression models.

ARC_MLPF="/gpfs/scratch/ehpc399/vincent/models/arc_properties_1M_2403/showers_df_evaluation/eval_full_1M_arc_*.pkl0_0_None.pt"
O5_MLPF="/gpfs/scratch/ehpc399/vincent/models/05_properties_1M_2403/showers_df_evaluation/eval_full_1M_05_*.pkl0_0_None.pt"
ARC_PANDORA="/gpfs/scratch/ehpc399/vincent/models/arc_properties_1M_2403/showers_df_evaluation/eval_full_1M_arc_*_pandora.pt"
O5_PANDORA="/gpfs/scratch/ehpc399/vincent/models/05_properties_1M_2403/showers_df_evaluation/eval_full_1M_05_*_pandora.pt"

DATATYPE="hitpf_pandora_compare"
OUTPUT_DIR="/gpfs/scratch/ehpc399/vincent/models/full_evaluation_compare_1M_arc_05"

CMD=(
  python -m src.evaluation.full_evaluation
  --arc-mlpf "${ARC_MLPF}"
  --o5-mlpf "${O5_MLPF}"
  --arc-pandora "${ARC_PANDORA}"
  --o5-pandora "${O5_PANDORA}"
  --datatype "${DATATYPE}"
  --output-dir "${OUTPUT_DIR}"
)

printf 'Running command:\n%s\n' "${CMD[*]}"
"${CMD[@]}"
