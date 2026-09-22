#!/bin/bash
# =============================================================================
# DELPHI FULL evaluation — stage-1 clustering + stage-2 EC/PID GRAFTED together.
# This is the configuration the model/DELANA comparison needs: stage 1 alone
# reports calorimeter deposits with tracks contributing exactly zero energy,
# and the stage-2 EC is the first thing that sees track_p.
#
#   MODEL=hitpf  EC_CKPT=... CLUST_CKPT=... sbatch scripts/bsc/eval_delphi_full.sh
#   MODEL=mask3d EC_CKPT=... CLUST_CKPT=... sbatch ...
#
# LOAD ORDER MATTERS — EC checkpoint first, clustering second, which overwrites
# the throwaway backbone the EC checkpoint carries (train_lightning1.py:236).
# Both loads are strict=False: watch the two `loaded N/M keys` lines.
# Event order in the output is NOT sorted-file order (--num-workers 4).
# =============================================================================
#SBATCH --job-name=delphi_eval_full
#SBATCH --output=/gpfs/scratch/ehpc1013/vriecher/slurm-logs/%x-%j.out
#SBATCH --error=/gpfs/scratch/ehpc1013/vriecher/slurm-logs/%x-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --qos=acc_debug
#SBATCH --account=ehpc1013

set -uo pipefail
PROJ=/gpfs/projects/ehpc1013/vriecher
SCR=/gpfs/scratch/ehpc1013/vriecher
PY=$PROJ/envs/mlpf-overlay/bin/python
REPO=$PROJ/delphi_study/mlpf
export WANDB_MODE=offline WANDB_DIR=$SCR
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NUMEXPR_MAX_THREADS=20 NUMEXPR_NUM_THREADS=20

MODEL=${MODEL:-hitpf}
case "$MODEL" in
  hitpf)  NETCFG=src/models/wrapper/example_mode_gatr_noise.py
          EXTRA=(--condensation -clust -clust_dim 3 --qmin 1 --use-average-cc-pos 0.99
                 --optimizer ranger)
          NETOPTS=() ;;
  mask3d) NETCFG=src/models/wrapper/example_mode_mask3d.py
          # --mask3d-use-mask-labels is MANDATORY here. Without it the EC
          # pipeline runs its own DPC clustering while inference uses
          # labels_from_masks, so the two disagree on how many clusters exist
          # and generate_showers_data_frame dies with e.g.
          #   RuntimeError: shape mismatch: value tensor of shape [44] cannot
          #   be broadcast to indexing result of shape [46]
          # (job 45227034). inference_oc.py:886 documents the requirement.
          EXTRA=(--optimizer adamW --mask3d-use-mask-labels)
          NETOPTS=(-o use_ipa_decoder False -o num_queries "${NUM_QUERIES:-320}"
                   -o window_size None) ;;
  *) echo "MODEL must be hitpf or mask3d" >&2; exit 2 ;;
esac

EC_CKPT=${EC_CKPT:?set EC_CKPT to the stage-2 checkpoint}
CLUST_CKPT=${CLUST_CKPT:?set CLUST_CKPT to the stage-1 checkpoint}
MODEL_DIR=${MODEL_DIR:-$SCR/trained-models/fulleval_${MODEL}}
RUN_NAME=${RUN_NAME:-fulleval_${MODEL}}
OUTPUT_NAME=${OUTPUT_NAME:-eval_full_${MODEL}.pkl}
VAL_DIR=${VAL_DIR:-$SCR/data/50k_validation_delana/digitized}
N_EVAL_FILES=${N_EVAL_FILES:-20}
mapfile -t ALL < <(ls -1 "$VAL_DIR"/pf_tree_*.parquet)
DATA_FILES=("${ALL[@]:0:$N_EVAL_FILES}")

mkdir -p "$MODEL_DIR/showers_df_evaluation" "$SCR/wandb"
cd "$REPO" || exit 1
echo "[full] model=$MODEL netcfg=$NETCFG"
echo "[full] EC    <- $EC_CKPT"
echo "[full] clust <- $CLUST_CKPT"
echo "[full] files : ${#DATA_FILES[@]}"

$PY -m src.train_lightning1 \
    --delphi --predict \
    --data-test "${DATA_FILES[@]}" \
    --name-output "$OUTPUT_NAME" \
    --data-config config_files/config_hits_track_delphi.yaml \
    --network-config "$NETCFG" \
    --model-prefix "$MODEL_DIR" \
    --load-model-weights "$EC_CKPT" \
    --load-model-weights-clustering "$CLUST_CKPT" \
    --correction --ec-model gatr-neutrals --regress-pos --regress-unit-p \
    --separate-PID-GATr --n-layers-PID-head 3 \
    --restrict_PID_charge --PID-4-class --add-track-chis \
    --num-workers 4 --gpus 0 --batch-size "${BATCH_SIZE:-1}" \
    --start-lr 1e-4 --num-epochs 1 --train-batches 1 \
    --fetch-step "${FETCH_STEP:-0.1}" \
    --log-wandb --wandb-displayname "$RUN_NAME" \
    --wandb-projectname mlpf-delphi --wandb-entity optimal-design \
    --tracks \
    "${EXTRA[@]}" "${NETOPTS[@]}"

RC=$?
echo "[full] exit=$RC"
ls -la "$MODEL_DIR/showers_df_evaluation/" 2>/dev/null | tail -3
exit $RC
