#!/bin/bash
#SBATCH --job-name=05_clustering_1M
#SBATCH --output=/gpfs/scratch/ehpc399/vincent/logs/%x-%j.out
#SBATCH --error=/gpfs/scratch/ehpc399/vincent/logs/%x-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:4
#SBATCH --time=3-00:00:00
#SBATCH --qos=acc_ehpc
#SBATCH --account=ehpc399

set -euo pipefail

BASE_NAME="05_clustering_1M"
RUN_TAG="$(date +%d%m)"
RUN_NAME="${BASE_NAME}_${RUN_TAG}"

ml miniforge/24.3.0-0
source "/gpfs/apps/MN5/ACC/MINIFORGE/24.3.0-0/etc/profile.d/conda.sh"
conda activate /gpfs/scratch/ehpc399/vincent/envs/HitPF

nvidia-smi

cd "/gpfs/scratch/ehpc399/vincent/code/mlpf"
pwd

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export SLURM_CPU_BIND=none
export TOKENIZERS_PARALLELISM=false

export WANDB_MODE=offline
export WANDB_PROJECT=mlpf_debug
export WANDB_ENTITY=ml4hep
export WANDB_DIR="/gpfs/scratch/ehpc399/vincent/wandb/${RUN_NAME}_${SLURM_JOB_ID}"
mkdir -p "${WANDB_DIR}"

DATA_DIR="/gpfs/scratch/ehpc399/vincent/data/1M_training/05/"
CFG_DATA="config_files/config_hits_track_v4.yaml"
CFG_NET="src/models/wrapper/example_mode_gatr_noise.py"
MODEL_PREFIX="/gpfs/scratch/ehpc399/vincent/models/${RUN_NAME}"
mkdir -p "${MODEL_PREFIX}"

CPUS=${SLURM_CPUS_PER_TASK:-20}
IFS="," read -r -a CUDA_IDS <<< "${CUDA_VISIBLE_DEVICES:-0}"
NGPU=${#CUDA_IDS[@]}
if [[ $NGPU -lt 1 ]]; then NGPU=1; fi

THREADS_PER_RANK=$(( (CPUS + NGPU - 1) / NGPU ))
if [[ $THREADS_PER_RANK -lt 1 ]]; then THREADS_PER_RANK=1; fi

export OMP_NUM_THREADS=${THREADS_PER_RANK}
export MKL_NUM_THREADS=${THREADS_PER_RANK}
export OPENBLAS_NUM_THREADS=${THREADS_PER_RANK}
export BLIS_NUM_THREADS=${THREADS_PER_RANK}
export NUMEXPR_MAX_THREADS=${THREADS_PER_RANK}
export NUMEXPR_NUM_THREADS=${THREADS_PER_RANK}
export KMP_AFFINITY=granularity=fine,compact,1,0
export KMP_BLOCKTIME=0

mkdir -p logs checkpoints

NUM_WORKERS=16

srun python -m src.train_lightning1 \
  --data-train "${DATA_DIR}" \
  --data-config "${CFG_DATA}" \
  --network-config "${CFG_NET}" \
  --model-prefix "${MODEL_PREFIX}/" \
  --num-workers "${NUM_WORKERS}" \
  --gpus 0,1,2,3 \
  --batch-size 20 \
  --start-lr 1e-3 \
  --num-epochs 10 \
  --optimizer ranger \
  --fetch-step 4 \
  --condensation \
  --log-wandb \
  --wandb-displayname "${RUN_NAME}" \
  --wandb-projectname mlpf_debug \
  --wandb-entity ml4hep \
  --frac_cluster_loss 0 \
  --qmin 3 \
  --use-average-cc-pos 0.98 \
  --tracks \
  --train-val-split 0.98 \
  --fetch-by-files \
  --train-batches 12250
