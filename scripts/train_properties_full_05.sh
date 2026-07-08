#!/bin/bash
#SBATCH --job-name=mlpf_props_dolores_05
#SBATCH --output=/gpfs/scratch/ehpc399/vincent/logs/%x-%j.out
#SBATCH --error=/gpfs/scratch/ehpc399/vincent/logs/%x-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1          # Lightning will spawn per-GPU workers
#SBATCH --cpus-per-task=80
#SBATCH --gres=gpu:4
#SBATCH --time=3-00:00:00
#SBATCH --qos=acc_ehpc
#SBATCH --account=ehpc399

set -euo pipefail

# --- Environment ---
ml miniforge/24.3.0-0
source "/gpfs/apps/MN5/ACC/MINIFORGE/24.3.0-0/etc/profile.d/conda.sh"
conda activate /gpfs/scratch/ehpc399/vincent/envs/HitPF

nvidia-smi

cd "/gpfs/scratch/ehpc399/vincent/code/mlpf"
pwd

# --- Runtime env ---
export CUDA_VISIBLE_DEVICES=0,1,2,3
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export SLURM_CPU_BIND=none
export TOKENIZERS_PARALLELISM=false

# --- Threading: split CPUs evenly across spawned GPU workers ---
CPUS=${SLURM_CPUS_PER_TASK:-80}
IFS=',' read -r -a CUDA_IDS <<< "${CUDA_VISIBLE_DEVICES:-0}"
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

# --- Weights & Biases ---
export WANDB_MODE=offline
export WANDB_PROJECT=mlpf_debug
export WANDB_ENTITY=ml4hep
export WANDB_DIR="/gpfs/scratch/ehpc399/vincent/wandb/${SLURM_JOB_ID}"
mkdir -p "${WANDB_DIR}"

# --- Paths / configs ---
DATA_DIR="/gpfs/scratch/ehpc399/vincent/data/500k_mix/05/"
CFG_DATA="config_files/config_hits_track_v4.yaml"
CFG_NET="src/models/wrapper/example_mode_gatr_noise.py"

MODEL_PREFIX="/gpfs/scratch/ehpc399/vincent/models/PROPS_500k_05_DoloresLike_$(date +%Y%m%d_%H%M%S)/"
mkdir -p "${MODEL_PREFIX}"

# --- Training setup (Dolores-like) ---
NUM_EPOCHS=100
TRAIN_BATCHES=6125
BATCH_SIZE=40
TRAIN_BATCHES_PER_GPU=$(( (TRAIN_BATCHES + NGPU - 1) / NGPU ))
NUM_WORKERS=$(( THREADS_PER_RANK / 2 ))
if [[ $NUM_WORKERS -lt 2 ]]; then NUM_WORKERS=2; fi

# --- Launch ---
python -m src.train_lightning1 \
  --data-train "${DATA_DIR}" \
  --data-config "${CFG_DATA}" \
  -clust \
  -clust_dim 3 \
  --network-config "${CFG_NET}" \
  --model-prefix "${MODEL_PREFIX}" \
  --wandb-displayname "E_PID_05_basicecor_v1_1_500k_4gpu" \
  --gpus 0,1,2,3 \
  --num-workers "${NUM_WORKERS}" \
  --batch-size "${BATCH_SIZE}" \
  --start-lr 1e-3 \
  --num-epochs "${NUM_EPOCHS}" \
  --optimizer ranger \
  --fetch-step 4 \
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
  --separate-PID-GATr \
  --n-layers-PID-head 3 \
  --fetch-by-files \
  --train-val-split 0.98 \
  --restrict_PID_charge \
  --PID-4-class \
  --balance-pid-classes \
  --train-batches "${TRAIN_BATCHES_PER_GPU}" \
  --use-gt-clusters
