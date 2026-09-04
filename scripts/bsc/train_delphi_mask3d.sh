#!/bin/bash
# =============================================================================
# DELPHI plain MASK3D (encoder + MaskFormerDecoder) — BSC MareNostrum 5.
#
# This is the recommended upstream variant ("train a mask3d model"), and it is
# NOT what scripts/bsc/smoke_delphi_attn_ipa.sh trains. The three are different
# models, not spellings of one:
#
#   Mask3D    InputNet+encoder + MaskFormerDecoder (iterative mask attention)
#             -> mask3d_model.ExampleWrapper, use_ipa_decoder=False   <-- THIS
#   GATr-IPA  GATr backbone + IPADecoder  (variant A/B)
#   Attn-IPA  attention backbone + IPADecoder (variant C)
#             -> attn_ipa_model.AttnIPAModel
#
# Use mask3d_model rather than the standalone variant files: it is the
# maintained superset (1248 lines / 78 kwargs vs 420 / 48) and the only one
# carrying the evaluation path, the EC adapter, ranger/lion — AND it can build
# the IPA decoder via `-o use_ipa_decoder True`, so training through it keeps
# the A/B open with checkpoint-compatible plumbing.
#
# SHARED KNOB WORTH KNOWING: both decoders seed queries with a FIXED
# `keys.norm(dim=-1)` topk (decoder.py:426). The learnable queryness head was
# removed because topk consumes only `.indices`, so it could never learn and it
# tripped DDP's unused-params check (job 40330284). Consequence: with
# num_queries=320 against ~56 truth objects/event, a large shower's many
# high-norm hits attract many seeds -> the shower fragments across queries.
# Measured on the 1-epoch Attn-IPA model: efficiency FALLS with energy
# (0.63 -> 0.42), <E_pred/E_true> = 0.33 on charged hadrons, 42 % fakes against
# HitPF's 4 %. Plain Mask3D shares this mechanism; it is not fixed for free.
# NUM_QUERIES is therefore the first knob to scan, not the last.
# =============================================================================
#SBATCH --job-name=delphi_mask3d
#SBATCH --output=/gpfs/scratch/ehpc1013/vriecher/slurm-logs/%x-%j.out
#SBATCH --error=/gpfs/scratch/ehpc1013/vriecher/slurm-logs/%x-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=80
#SBATCH --gres=gpu:4
#SBATCH --time=10:00:00
#SBATCH --qos=acc_ehpc
#SBATCH --account=ehpc1013

set -uo pipefail
PROJ=/gpfs/projects/ehpc1013/vriecher
SCR=/gpfs/scratch/ehpc1013/vriecher
PY=$PROJ/envs/mlpf-overlay/bin/python
REPO=$PROJ/delphi_study/mlpf

export WANDB_MODE=offline WANDB_DIR=$SCR
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True SLURM_CPU_BIND=none
export OMP_NUM_THREADS=20 MKL_NUM_THREADS=20
export NUMEXPR_MAX_THREADS=20 NUMEXPR_NUM_THREADS=20

GPUS_PER_NODE=${GPUS_PER_NODE:-4}
GPU_LIST=${GPU_LIST:-0,1,2,3}
MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
MASTER_PORT=${MASTER_PORT:-29500}
RUN_NAME=${RUN_NAME:-delphi_mask3d}
NUM_QUERIES=${NUM_QUERIES:-320}
TRACK_LOSS_WEIGHT=${TRACK_LOSS_WEIGHT:-3.0}
WINDOW_SIZE=${WINDOW_SIZE:-None}          # None => xformers, not flash-attn
if [[ "${USE_AMP:-1}" == "0" ]]; then AMP_FLAG=""; else AMP_FLAG="--use-amp"; fi
NETWORK_OPTS=${NETWORK_OPTS:-}

mkdir -p $SCR/wandb $SCR/slurm-logs $SCR/trained-models/${RUN_NAME}
cd "$REPO" || exit 1
echo "[mask3d] host=$(hostname) gpus=$GPUS_PER_NODE list=$GPU_LIST"
echo "[mask3d] run=$RUN_NAME epochs=${NUM_EPOCHS:-1} train_batches=${TRAIN_BATCHES:-50} batch=${BATCH_SIZE:-64}"
echo "[mask3d] num_queries=$NUM_QUERIES track_loss_weight=$TRACK_LOSS_WEIGHT window_size=$WINDOW_SIZE amp='${AMP_FLAG:-off}' opts='$NETWORK_OPTS'"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

GPU_CSV=$SCR/slurm-logs/${SLURM_JOB_NAME}-${SLURM_JOB_ID}.gpu.csv
if [[ "${GPU_MONITOR:-1}" == "1" ]]; then
    nvidia-smi --query-gpu=index,utilization.gpu,memory.used \
               --format=csv,noheader,nounits -l 1 > "$GPU_CSV" 2>/dev/null &
    GPU_MON_PID=$!; trap 'kill $GPU_MON_PID 2>/dev/null' EXIT
fi

srun --ntasks="$SLURM_NNODES" --ntasks-per-node=1 \
  $PY -m torch.distributed.run \
    --nnodes="$SLURM_NNODES" --nproc_per_node="$GPUS_PER_NODE" \
    --master_addr="$MASTER_ADDR" --master_port="$MASTER_PORT" \
    --rdzv_backend=c10d --rdzv_endpoint="$MASTER_ADDR:$MASTER_PORT" \
    --rdzv_id="$SLURM_JOB_ID" --node_rank="$SLURM_PROCID" \
    -m src.train_lightning1 \
      --delphi \
      --data-train "$SCR/data/500k_delana/digitized/" \
      --data-config config_files/config_hits_track_delphi.yaml \
      --network-config src/models/wrapper/example_mode_mask3d.py \
      --model-prefix "$SCR/trained-models/${RUN_NAME}" \
      --num-workers "${NUM_WORKERS:-12}" \
      --prefetch-factor "${PREFETCH_FACTOR:-8}" \
      --gpus "$GPU_LIST" \
      --batch-size "${BATCH_SIZE:-64}" \
      --start-lr "${START_LR:-1e-4}" \
      --num-epochs "${NUM_EPOCHS:-1}" \
      --optimizer adamW \
      --fetch-by-files --fetch-step 4 \
      $AMP_FLAG \
      --gradient-clip-val "${GRAD_CLIP:-0.1}" \
      --log-wandb --wandb-displayname "${RUN_NAME}" \
      --wandb-projectname mlpf-delphi --wandb-entity optimal-design \
      --tracks --train-val-split 0.98 \
      ${SEED:+--seed "$SEED"} \
      --train-batches "${TRAIN_BATCHES:-50}" \
      -o use_ipa_decoder False \
      -o num_queries "$NUM_QUERIES" \
      -o track_loss_weight "$TRACK_LOSS_WEIGHT" \
      -o window_size "$WINDOW_SIZE" \
      $NETWORK_OPTS

RC=$?
echo "[mask3d] exit=$RC"
exit $RC
