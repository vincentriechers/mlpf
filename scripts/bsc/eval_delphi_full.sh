#!/bin/bash
# =============================================================================
# DELPHI FULL evaluation — stage-1 clustering + stage-2 EC/PID GRAFTED together.
#
# This is the configuration the four-way comparison actually needs. Stage-1
# alone gives raw CALORIMETER DEPOSITS: tracks contribute exactly zero energy
# (dataclasses.py:123 `e_tracks = X_track[:,5]*0`, summed at
# inference_oc.py:705), so charged particles — 55 % of the visible energy — are
# measured by the calorimeter instead of by their track. DELANA uses track
# momentum. The stage-2 EC is the first thing that sees `track_p`
# (post_clustering_features.py:47), which is why only the grafted model can
# plausibly beat DELANA.
#
# The graft is done by train_lightning1.py:236 (`_netcfg_aware` branch):
#   --load-model-weights            <stage-2 EC/PID checkpoint>   loaded FIRST,
#                                    shape-matched keys only
#   --load-model-weights-clustering <stage-1 clustering ckpt>     overwrites the
#                                    clustering backbone afterwards
# Order matters: the EC checkpoint carries its own throwaway clustering backbone
# (trained on GT clusters) which must be replaced by the real one.
#
#   MODEL=hitpf  EC_CKPT=... CLUST_CKPT=... sbatch scripts/bsc/eval_delphi_full.sh
#   MODEL=mask3d EC_CKPT=... CLUST_CKPT=... sbatch ...
#
# WATCH THE TWO LOAD LINES. load_test_model prints how many keys each checkpoint
# contributed; the EC line should report ~0 EC-head keys skipped. It loads with
# strict=False, so a mismatch is silent.
#
# EVENT ORDER IN THE OUTPUT DATAFRAME IS NOT SORTED-FILE ORDER.
# `--num-workers 4` makes the DataLoader round-robin one event from each of four
# open parquets, so `number_batch` indexes the STREAM, not the file list:
#   stream i  ->  file (i%4 + 4*((i//4)//100)), event ((i//4)%100)
# Verified on the HitPF and Mask3D full evals, 4800/4800 events. Any analysis
# that joins this dataframe to the parquets positionally (truth, DELANA) is then
# comparing each event against a DIFFERENT event, which inflated every model
# sigma68 in this project by ~32% (HitPF read 0.201 instead of 0.152).
# `analysis/three_way_visible_energy.py` now recovers and verifies the
# permutation itself, so existing dataframes are usable as they are. Setting
# --num-workers 1 here would also fix it, at ~4x the wall clock.
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
VAL_DIR=$SCR/data/50k_validation_delana/digitized
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
