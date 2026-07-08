#!/bin/bash
#SBATCH --job-name=bash
#SBATCH --output=test-job.out
#SBATCH --error=test-job.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1          # Lightning will spawn per-GPU workers
#SBATCH --cpus-per-task=80
#SBATCH --gres=gpu:4
#SBATCH --time=00:15:00
#SBATCH --qos=acc_ehpc
#SBATCH --account=ehpc399

set -euo pipefail

# --- Environment ---
ml miniforge/24.3.0-0
source "/gpfs/apps/MN5/ACC/MINIFORGE/24.3.0-0/etc/profile.d/conda.sh"
conda activate /gpfs/scratch/ehpc399/jp/envs/mlpf

nvidia-smi

cd "/gpfs/scratch/ehpc399/dolo/repos_dolo/mlpf"
pwd

# --- Runtime env ---
export CUDA_VISIBLE_DEVICES=0,1,2,3
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export SLURM_CPU_BIND=none
# export CUDA_LAUNCH_BLOCKING=1     # uncomment only for debugging
export TOKENIZERS_PARALLELISM=false

# --- Weights & Biases (offline at BSC) ---
export WANDB_MODE=offline
export WANDB_PROJECT=mlpf_debug
export WANDB_ENTITY=ml4hep
export WANDB_DIR=/gpfs/scratch/ehpc399/dolo/wandb/${SLURM_JOB_ID}
mkdir -p "${WANDB_DIR}"

# --- Paths / configs (edit if needed) ---
DATA_DIR="/gpfs/scratch/ehpc399/dolo/MLPF_data/gun_dr_logE_211125/"
CFG_DATA="config_files/config_hits_track_v4.yaml"
CFG_NET="src/models/wrapper/example_mode_gatr_noise.py"
MODEL_PREFIX="/gpfs/scratch/ehpc399/dolo/models_trained_CLD/ALLEGRO_gun_dr05_v3/"
mkdir -p "${MODEL_PREFIX}"

# --- Threading: split CPUs evenly across spawned GPU workers ---
CPUS=${SLURM_CPUS_PER_TASK:-80}
# Count GPUs from CUDA_VISIBLE_DEVICES (fallback to 1)
IFS=',' read -r -a CUDA_IDS <<< "${CUDA_VISIBLE_DEVICES:-0}"
NGPU=${#CUDA_IDS[@]}
if [[ $NGPU -lt 1 ]]; then NGPU=1; fi

# ceil(CPUS / NGPU)
THREADS_PER_RANK=$(( (CPUS + NGPU - 1) / NGPU ))
if [[ $THREADS_PER_RANK -lt 1 ]]; then THREADS_PER_RANK=1; fi

# BLAS / OpenMP threads per spawned process
export OMP_NUM_THREADS=${THREADS_PER_RANK}
export MKL_NUM_THREADS=${THREADS_PER_RANK}
export OPENBLAS_NUM_THREADS=${THREADS_PER_RANK}
export BLIS_NUM_THREADS=${THREADS_PER_RANK}

# numexpr MUST be set before any import that could load it
export NUMEXPR_MAX_THREADS=${THREADS_PER_RANK}
export NUMEXPR_NUM_THREADS=${THREADS_PER_RANK}

# Optional: avoid PyTorch interop oversubscription
export KMP_AFFINITY=granularity=fine,compact,1,0
export KMP_BLOCKTIME=0

# --- I/O / logs ---
mkdir -p logs checkpoints

# --- Launch (Lightning spawns one worker per GPU inside this task) ---
# Keep DataLoader workers modest per rank
NUM_WORKERS=$(( THREADS_PER_RANK / 2 ))
if [[ $NUM_WORKERS -lt 2 ]]; then NUM_WORKERS=2; fi
# --- Launch (Lightning spawns one worker per GPU) ---
python -m src.train_lightning1 \
    --data-train "${DATA_DIR}" \
    --data-config "${CFG_DATA}" \
    --network-config "${CFG_NET}" \
    --model-prefix "${MODEL_PREFIX}/" \
    --num-workers 16 \
    --gpus 0,1,2,3 \
    --batch-size 20 \
    --start-lr 1e-3 \
    --num-epochs 20 \
    --optimizer ranger \
    --fetch-step 1 \
    --condensation \
    --log-wandb \
    --wandb-displayname ALLEGRO_v0_1411 \
    --wandb-projectname mlpf_debug \
    --wandb-entity ml4hep \
    --frac_cluster_loss 0 \
    --qmin 3 \
    --use-average-cc-pos 0.98 \
    --tracks \
    --train-val-split 0.98 \
    --fetch-by-files \
    --train-batches 12500 \
    --allegro 

