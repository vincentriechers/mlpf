#!/bin/bash
#SBATCH --job-name=mlpf_eval_full_500k_arc
#SBATCH --output=/gpfs/scratch/ehpc399/vincent/logs/%x-%j.out
#SBATCH --error=/gpfs/scratch/ehpc399/vincent/logs/%x-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=80
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --qos=acc_ehpc
#SBATCH --account=ehpc399

set -euo pipefail
shopt -s nullglob

ml miniforge/24.3.0-0
CONDA_BASE="$(dirname "$(dirname "$(which conda)")")"
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate /gpfs/scratch/ehpc399/vincent/envs/HitPF
PYTHON_BIN="/gpfs/scratch/ehpc399/vincent/envs/HitPF/bin/python"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Expected python not found: ${PYTHON_BIN}"
  exit 1
fi

echo "Conda executable: $(which conda)"
echo "Python executable in PATH: $(which python)"
echo "CONDA_PREFIX: ${CONDA_PREFIX:-<unset>}"
"${PYTHON_BIN}" -c "import sys, torch; print(sys.executable); print(torch.__version__)"

cd /gpfs/scratch/ehpc399/vincent/code/mlpf
nvidia-smi
pwd

export WANDB_MODE=offline
export WANDB_PROJECT=mlpf_debug_eval
export WANDB_ENTITY=ml4hep
export WANDB_DIR="/gpfs/scratch/ehpc399/vincent/wandb/${SLURM_JOB_ID}"
mkdir -p "${WANDB_DIR}"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export SLURM_CPU_BIND=none
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export BLIS_NUM_THREADS=8
export NUMEXPR_MAX_THREADS=64
export NUMEXPR_NUM_THREADS=8

CFG_DATA="config_files/config_hits_track_v4.yaml"
CFG_NET="src/models/wrapper/example_mode_gatr_noise.py"
CLUSTER_MODEL_DIR="/gpfs/scratch/ehpc399/vincent/models/arc_clustering_1M_2303"
PROPS_MODEL_DIR="/gpfs/scratch/ehpc399/vincent/models/arc_properties_1M_2403"
OUTPUT_DIR="${PROPS_MODEL_DIR}"

CLUSTER_CKPT="$(ls -1t "${CLUSTER_MODEL_DIR}"/*.ckpt 2>/dev/null | head -n1 || true)"
if [[ -z "${CLUSTER_CKPT}" ]]; then
  echo "No clustering checkpoint found in ${CLUSTER_MODEL_DIR}"
  exit 1
fi

PROPS_CKPT="$(ls -1t "${PROPS_MODEL_DIR}"/*.ckpt 2>/dev/null | head -n1 || true)"
if [[ -z "${PROPS_CKPT}" ]]; then
  echo "No properties checkpoint found in ${PROPS_MODEL_DIR}"
  exit 1
fi

mkdir -p "${OUTPUT_DIR}/showers_df_evaluation"

echo "Using clustering checkpoint: ${CLUSTER_CKPT}"
echo "Using properties checkpoint: ${PROPS_CKPT}"
echo "Writing outputs to: ${OUTPUT_DIR}/showers_df_evaluation/"

declare -A DATASETS=(
  [eCH]="/gpfs/scratch/ehpc399/vincent/data/1M_training/eval/eCH/arc/*.parquet"
  [neutral]="/gpfs/scratch/ehpc399/vincent/data/1M_training/eval/neutral/arc/*.parquet"
)

for SPLIT in eCH neutral; do
  DATA_PATTERN="${DATASETS[$SPLIT]}"
  DATA_FILES=(${DATA_PATTERN})
  OUTPUT_NAME="eval_full_1M_arc_${SPLIT}.pkl"
  DISPLAY_NAME="eval_full_1M_arc_${SPLIT}"

  if [[ ${#DATA_FILES[@]} -eq 0 ]]; then
    echo "No parquet files found for ${SPLIT} with pattern: ${DATA_PATTERN}"
    exit 1
  fi

  echo "Running ARC full eval for ${SPLIT} with ${#DATA_FILES[@]} files"

  "${PYTHON_BIN}" -m src.train_lightning1 \
    --predict \
    --data-test "${DATA_FILES[@]}" \
    --name-output "${OUTPUT_NAME}" \
    --data-config "${CFG_DATA}" \
    -clust \
    -clust_dim 3 \
    --network-config "${CFG_NET}" \
    --model-prefix "${OUTPUT_DIR}" \
    --load-model-weights-clustering "${CLUSTER_CKPT}" \
    --load-model-weights "${PROPS_CKPT}" \
    --wandb-displayname "${DISPLAY_NAME}" \
    --num-workers 16 \
    --gpus 0 \
    --batch-size 1 \
    --start-lr 1e-3 \
    --num-epochs 100 \
    --optimizer ranger \
    --fetch-step 1 \
    --condensation \
    --log-wandb \
    --wandb-projectname "${WANDB_PROJECT}" \
    --wandb-entity "${WANDB_ENTITY}" \
    --frac_cluster_loss 0 \
    --qmin 1 \
    --use-average-cc-pos 0.99 \
    --lr-scheduler reduceplateau \
    --tracks \
    --correction \
    --ec-model gatr-neutrals \
    --regress-pos \
    --add-track-chis \
    --freeze-clustering \
    --regress-unit-p \
    --n-layers-PID-head 3 \
    --separate-PID-GATr \
    --fetch-by-files \
    --restrict_PID_charge \
    --PID-4-class \
    --pandora
done
