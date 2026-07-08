#!/bin/bash
set -euo pipefail

# Copy this script to AFS and submit it from there.
# Usage:
#   ./submit_tracking_performance_arc_05_condor.sh 20000
# If MAX_EVENTS is larger than the available statistics, the plotting script
# simply processes all available events.

MAX_EVENTS="${1:-20000}"
MLPF_PLOT_USETEX="${MLPF_PLOT_USETEX:-auto}"
RUN_PREFLIGHT="${RUN_PREFLIGHT:-1}"

AFS_BASE="/afs/cern.ch/user/v/vriecher/private/tracking_plots_arc_05_condor"
EOS_PLOT_SCRIPT="/eos/home-v/vriecher/mlpf_arc/mlpf/scripts/scripts_vincent/plot_tracking_performance_arc_05.py"
EOS_ENV_SETUP_SCRIPT="/eos/home-v/vriecher/scripts/setup_tmp_env.sh"
CLD_DIR="/eos/experiment/fcc/users/m/mgarciam/mlpf/CLD/train/Z_ss_CLD_o2_v05/05"
ARC_DIR="/eos/experiment/fcc/users/m/mgarciam/mlpf/CLD/train/Z_ss_CLD_o2_v05/arc"
OUTPUT_DIR="/eos/home-v/vriecher/mlpf_arc/mlpf/tracking_plots_arc_05_condor"

REQUEST_CPUS="2"
REQUEST_MEMORY="8 GB"
REQUEST_DISK="4 GB"

RUN_SCRIPT="${AFS_BASE}/run_tracking_performance_arc_05_condor.sh"
SUBMIT_FILE="${AFS_BASE}/submit_tracking_performance_arc_05_condor.sub"
LOG_DIR="${AFS_BASE}/logs"
MINIFORGE_DIR="/tmp/vriecher/miniforge3"
ENV_PREFIX="/tmp/vriecher/envs/pytorch_cpuOnly"
PYTHON_BIN="${ENV_PREFIX}/bin/python"

mkdir -p "${AFS_BASE}" "${LOG_DIR}" "${OUTPUT_DIR}"

cat > "${RUN_SCRIPT}" <<EOF
#!/bin/bash
set -euo pipefail

MAX_EVENTS="\${MAX_EVENTS:-}"
MLPF_PLOT_USETEX="\${MLPF_PLOT_USETEX:-auto}"
ENV_SETUP_SCRIPT="${EOS_ENV_SETUP_SCRIPT}"
PLOT_SCRIPT="${EOS_PLOT_SCRIPT}"
CLD_DIR="${CLD_DIR}"
ARC_DIR="${ARC_DIR}"
OUTPUT_DIR="${OUTPUT_DIR}"
MINIFORGE_DIR="/tmp/vriecher/miniforge3"
ENV_PREFIX="/tmp/vriecher/envs/pytorch_cpuOnly"
PYTHON_BIN="\${ENV_PREFIX}/bin/python"

if [ ! -f "\${ENV_SETUP_SCRIPT}" ]; then
  echo "Env setup script not found: \${ENV_SETUP_SCRIPT}" >&2
  exit 1
fi

echo "Preparing tmp env via: \${ENV_SETUP_SCRIPT}"
bash "\${ENV_SETUP_SCRIPT}"

if [ ! -f "\${MINIFORGE_DIR}/etc/profile.d/conda.sh" ]; then
  echo "conda.sh not found after env setup: \${MINIFORGE_DIR}/etc/profile.d/conda.sh" >&2
  exit 1
fi

source "\${MINIFORGE_DIR}/etc/profile.d/conda.sh"
conda activate "\${ENV_PREFIX}"

if [ ! -x "\${PYTHON_BIN}" ]; then
  echo "Python not found after env setup: \${PYTHON_BIN}" >&2
  exit 1
fi

echo "TeX tool check on worker:"
for tool in latex pdflatex dvipng gs; do
  echo "  \${tool}: \$(command -v "\${tool}" || echo MISSING)"
done

if command -v latex >/dev/null 2>&1 && command -v dvipng >/dev/null 2>&1 && command -v gs >/dev/null 2>&1; then
  if [ "\${MLPF_PLOT_USETEX}" = "auto" ]; then
    export MLPF_PLOT_USETEX=1
  else
    export MLPF_PLOT_USETEX
  fi
else
  export MLPF_PLOT_USETEX=0
fi

echo "Using MLPF_PLOT_USETEX=\${MLPF_PLOT_USETEX}"

CMD=(
  "\${PYTHON_BIN}"
  "\${PLOT_SCRIPT}"
  --cld-dir "\${CLD_DIR}"
  --arc-dir "\${ARC_DIR}"
  --output-dir "\${OUTPUT_DIR}"
)

if [ -n "\${MAX_EVENTS}" ] && [ "\${MAX_EVENTS}" != "all" ] && [ "\${MAX_EVENTS}" != "ALL" ]; then
  CMD+=(--max-events "\${MAX_EVENTS}")
fi

echo "Running: \${CMD[*]}"
"\${CMD[@]}"
EOF

chmod +x "${RUN_SCRIPT}"

if [ "${RUN_PREFLIGHT}" != "0" ]; then
  echo "Running local preflight smoke test before condor submission..."
  bash "${EOS_ENV_SETUP_SCRIPT}"
  source "${MINIFORGE_DIR}/etc/profile.d/conda.sh"
  conda activate "${ENV_PREFIX}"
  if [ "${MLPF_PLOT_USETEX}" = "auto" ]; then
    export MLPF_PLOT_USETEX
  else
    export MLPF_PLOT_USETEX
  fi
  "${PYTHON_BIN}" "${EOS_PLOT_SCRIPT}" --output-dir "${OUTPUT_DIR}" --smoke-test
fi

cat > "${SUBMIT_FILE}" <<EOF
universe              = vanilla
executable            = ${RUN_SCRIPT}
environment           = "MAX_EVENTS=${MAX_EVENTS} MLPF_PLOT_USETEX=${MLPF_PLOT_USETEX}"

output                = ${LOG_DIR}/tracking_performance.\$(ClusterId).\$(ProcId).out
error                 = ${LOG_DIR}/tracking_performance.\$(ClusterId).\$(ProcId).err
log                   = ${LOG_DIR}/tracking_performance.\$(ClusterId).log

request_cpus          = ${REQUEST_CPUS}
request_memory        = ${REQUEST_MEMORY}
request_disk          = ${REQUEST_DISK}

queue
EOF

echo "Submitting Condor job with MAX_EVENTS=${MAX_EVENTS} and MLPF_PLOT_USETEX=${MLPF_PLOT_USETEX}"
condor_submit "${SUBMIT_FILE}"
