#!/bin/bash
#SBATCH --job-name=mlpf_eval_full_500k_05
#SBATCH --output=/gpfs/scratch/ehpc399/vincent/logs/%x-%j.out
#SBATCH --error=/gpfs/scratch/ehpc399/vincent/logs/%x-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=80
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
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

# Adjust DATA_TEST or model directories if you want to evaluate a different sample/checkpoints.
DATA_TEST="/gpfs/scratch/ehpc399/vincent/data/500k_mix/evaluation/evaluation_eCH/05/pf_tree_2.parquet"
CFG_DATA="config_files/config_hits_track_v4.yaml"
CFG_NET="src/models/wrapper/example_mode_gatr_noise.py"
CLUSTER_MODEL_DIR="/gpfs/scratch/ehpc399/vincent/models/500k_05"
PROPS_MODEL_DIR="/gpfs/scratch/ehpc399/vincent/models/PROPS_500k_05_DoloresLike_20260309_130956"
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

"${PYTHON_BIN}" -m src.train_lightning1 \
  --predict \
  --data-test "${DATA_TEST}" \
  --name-output "eval_full_500k_05.pkl" \
  --data-config "${CFG_DATA}" \
  -clust \
  -clust_dim 3 \
  --network-config "${CFG_NET}" \
  --model-prefix "${OUTPUT_DIR}" \
  --load-model-weights-clustering "${CLUSTER_CKPT}" \
  --load-model-weights "${PROPS_CKPT}" \
  --wandb-displayname "eval_full_500k_05" \
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
  --PID-4-class
