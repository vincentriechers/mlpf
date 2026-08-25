#!/bin/bash
# =============================================================================
# DELPHI HitPF (object-condensation clustering) — BSC MareNostrum 5 SMOKE RUN.
#
# Purpose: prove the whole chain (env -> data -> loader -> model -> loss -> ckpt)
# on a handful of batches before spending from the 20 750 node-hour allocation.
# Runs on acc_debug (2 h cap, priority 10000) so it jumps the queue.
#
# Scale up with:  sbatch --qos=acc_ehpc --time=48:00:00 --gres=gpu:4 ...
# =============================================================================
#SBATCH --job-name=delphi_hitpf_smoke
#SBATCH --output=/gpfs/scratch/ehpc1013/vriecher/slurm-logs/%x-%j.out
#SBATCH --error=/gpfs/scratch/ehpc1013/vriecher/slurm-logs/%x-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --qos=acc_debug
#SBATCH --account=ehpc1013

set -uo pipefail

PROJ=/gpfs/projects/ehpc1013/vriecher
SCR=/gpfs/scratch/ehpc1013/vriecher
PY=$PROJ/envs/mlpf-overlay/bin/python        # BSC mlpf env + cu118 xformers
REPO=$PROJ/delphi_study/mlpf

# Compute nodes have NO outbound internet -> W&B must be offline, synced later
# from a transfer node with:  wandb sync --include-offline <dir>
export WANDB_MODE=offline
export WANDB_DIR=$SCR/wandb
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export SLURM_CPU_BIND=none

GPUS_PER_NODE=${GPUS_PER_NODE:-1}
MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
MASTER_PORT=${MASTER_PORT:-29500}

RUN_NAME=${RUN_NAME:-delphi_hitpf_smoke}
mkdir -p $SCR/wandb $SCR/trained-models $SCR/slurm-logs

cd "$REPO" || exit 1
echo "[smoke] host=$(hostname) gpus=$GPUS_PER_NODE python=$PY"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

srun --ntasks="$SLURM_NNODES" --ntasks-per-node=1 \
  $PY -m torch.distributed.run \
    --nnodes="$SLURM_NNODES" \
    --nproc_per_node="$GPUS_PER_NODE" \
    --master_addr="$MASTER_ADDR" \
    --master_port="$MASTER_PORT" \
    --rdzv_backend=c10d \
    --rdzv_endpoint="$MASTER_ADDR:$MASTER_PORT" \
    --rdzv_id="$SLURM_JOB_ID" \
    --node_rank="$SLURM_PROCID" \
    -m src.train_lightning1 \
      --delphi \
      --data-train "$SCR/data/500k_delana/digitized/" \
      --data-config config_files/config_hits_track_delphi.yaml \
      --network-config src/models/wrapper/example_mode_gatr_noise.py \
      --model-prefix "$SCR/trained-models/${RUN_NAME}" \
      --num-workers 4 \
      --gpus 0 \
      --batch-size "${BATCH_SIZE:-20}" \
      --start-lr "${START_LR:-2.5e-4}" \
      --num-epochs 1 \
      --optimizer ranger \
      --fetch-by-files \
      --fetch-step 4 \
      --condensation \
      --log-wandb \
      --wandb-displayname "${RUN_NAME}" \
      --wandb-projectname mlpf-delphi \
      --wandb-entity optimal-design \
      --frac_cluster_loss 0 \
      --qmin 3 \
      --use-average-cc-pos 0.98 \
      --tracks \
      --train-val-split 0.98 \
      --beta-noise-weight "${BETA_NOISE_WEIGHT:-1.0}" \
      --beta-noise-weight-track "${BETA_NOISE_WEIGHT_TRACK:-0.05}" \
      --train-batches "${TRAIN_BATCHES:-50}"

echo "[smoke] exit=$?"
