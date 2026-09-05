#!/bin/bash
# =============================================================================
# DELPHI DPC knob scan — MANY POINTS IN ONE JOB.
#
# `acc_debug` allows exactly one job per user, so a one-point-per-job scan is
# serialised by the queue at ~6 min of queue+startup per point. This runs the
# whole grid inside a single allocation instead: the model and the parquets are
# re-read per point (the eval entrypoint is not re-enterable), but the queue
# wait is paid once.
#
# The knobs live in `inference_oc.DPC_custom_CLD` and are read from the
# environment, so this is POST-HOC on an existing checkpoint — no retraining.
#
#   POINTS='label:KEY=VAL[,KEY=VAL...] ...' sbatch scripts/bsc/scan_dpc_grid.sh
#
# Each point is a label plus the environment to set for it, so this scans any
# post-hoc knob, not just the DPC ones. The legacy form `label:delta:rho` is
# still accepted and means DPC_DELTA_MIN/DPC_RHO_MIN.
#
#   POINTS='base:DPC_DELTA_MIN=0.10 nobtr:DPC_DELTA_MIN=0.10,BAD_TRACK_REMOVAL=0'
#
# Env: POINTS, MODEL_DIR, CKPT, N_EVAL_FILES, TAG, DEBUG_RHO_FIRST, EXTRA_ENV, MODE
#
# MODE=clust (default) evaluates stage 1 alone -- fast (~6 min / 20 files), and
# the right mode for anything that only changes CLUSTERING. MODE=full grafts the
# stage-2 EC/PID head on (EC_CKPT + CLUST_CKPT, ~13 min / 20 files) and is the
# only mode that can see a TRACK effect at all: stage-1 energy is calorimeter
# deposits only (`dataclasses.py:123` zeroes `e_tracks`), so a knob that changes
# which tracks land in which cluster moves nothing in a MODE=clust sigma68.
#
# Why rho_min is worth a decade scan and not a +/-2x one:
#   local_density_energy sums e_hits inside d_c, so `rho` carries GeV. The
#   0.05 default is therefore an ABSOLUTE energy-density cut inherited from CLD,
#   whose cells are far more energetic than DELPHI's ~15 MeV ones. Set
#   DEBUG_RHO_FIRST=1 to print the per-event rho percentiles for the first point
#   and see where 0.05 actually falls.
# =============================================================================
#SBATCH --job-name=dpc_grid
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

MODE=${MODE:-clust}
MODEL=${MODEL:-hitpf}
# The EC/PID flag block is identical for both models; only the network config,
# the checkpoints and the model-specific clustering flags differ. Mirrors
# scripts/bsc/eval_delphi_full.sh exactly -- check the two `loaded ... keys`
# lines in the log against a reference run before trusting any number.
EC_ARGS=(--correction --ec-model gatr-neutrals --regress-pos --regress-unit-p
         --separate-PID-GATr --n-layers-PID-head 3
         --restrict_PID_charge --PID-4-class --add-track-chis)
case "$MODEL" in
  hitpf)
    NETCFG=src/models/wrapper/example_mode_gatr_noise.py
    DEF_CLUST=$SCR/trained-models/hitpf_stage1/_epoch=16_step=104000.ckpt
    DEF_EC=$SCR/trained-models/hitpf_stage2_bal/_epoch=1_step=12000.ckpt
    DEF_FULLDIR=$SCR/trained-models/fulleval_hitpf
    DEF_CLUSTDIR=$SCR/trained-models/hitpf_stage1
    FULL_FLAGS=(--condensation -clust -clust_dim 3 --qmin 1 --use-average-cc-pos 0.99
                --optimizer ranger)
    CLUST_FLAGS=(--condensation -clust -clust_dim 3 --qmin 3 --use-average-cc-pos 0.98
                 --optimizer ranger)
    NETOPTS=() ;;
  mask3d)
    NETCFG=src/models/wrapper/example_mode_mask3d.py
    DEF_CLUST=$SCR/trained-models/mask3d_stage1/_epoch=15_step=30500.ckpt
    DEF_EC=$SCR/trained-models/mask3d_stage2_bal/_epoch=1_step=12000.ckpt
    DEF_FULLDIR=$SCR/trained-models/fulleval_mask3d
    DEF_CLUSTDIR=$SCR/trained-models/mask3d_stage1
    # --mask3d-use-mask-labels is MANDATORY: without it the EC pipeline runs its
    # own DPC clustering while inference uses labels_from_masks, the two
    # disagree on the cluster count, and generate_showers_data_frame dies.
    FULL_FLAGS=(--optimizer adamW --mask3d-use-mask-labels)
    CLUST_FLAGS=(--optimizer adamW)
    NETOPTS=(-o use_ipa_decoder False -o num_queries "${NUM_QUERIES:-320}"
             -o window_size None) ;;
  attnipa)
    # Attn-IPA checkpoints are evaluated through mask3d_model with
    # use_ipa_decoder True -- attn_ipa_model has no eval path, and the two are
    # deliberately state-dict compatible. EVERY SHAPE-BEARING TRAINING OPTION
    # must be repeated here or load_test_model keeps only the shape-matching
    # keys (strict=False) and evaluates a half-random model. WATCH THE LOAD LINE.
    #
    # Eval parses `-o` values with a bare ast.literal_eval, unlike training which
    # falls back to the raw string, so string values must be QUOTED literals:
    # "'kmeans'" not kmeans.
    NETCFG=src/models/wrapper/example_mode_mask3d.py
    DEF_CLUST=$SCR/trained-models/attnipa_ref/_epoch=7_step=97500.ckpt
    DEF_EC=$SCR/trained-models/mask3d_stage2_bal/_epoch=1_step=12000.ckpt
    DEF_FULLDIR=$SCR/trained-models/fulleval_attnipa
    DEF_CLUSTDIR=$SCR/trained-models/attnipa_ref
    FULL_FLAGS=(--optimizer adamW --mask3d-use-mask-labels)
    CLUST_FLAGS=(--optimizer adamW)
    NETOPTS=(-o use_ipa_decoder True
             -o num_queries "${NUM_QUERIES:-320}"
             -o window_size None
             -o cross_attn_mode "'kmeans'"
             -o kmeans_value_proj True
             -o ipa_mask_direction_aware True
             -o dynamic_query_source "'encoder'"
             -o track_seed_bonus 6e4
             -o track_loss_weight 3.0) ;;
  *) echo "MODEL must be hitpf, mask3d or attnipa, got '$MODEL'" >&2; exit 2 ;;
esac
if [[ "$MODE" == "full" ]]; then
    MODEL_DIR=${MODEL_DIR:-$DEF_FULLDIR}
    EC_CKPT=${EC_CKPT:-$DEF_EC}
    CKPT=${CKPT:-$DEF_CLUST}
    [[ -f "$EC_CKPT" ]] || { echo "no EC checkpoint at $EC_CKPT" >&2; exit 2; }
    MODE_ARGS=(--load-model-weights "$EC_CKPT" --load-model-weights-clustering "$CKPT"
               "${EC_ARGS[@]}" "${FULL_FLAGS[@]}")
else
    MODEL_DIR=${MODEL_DIR:-$DEF_CLUSTDIR}
    CKPT=${CKPT:-$DEF_CLUST}
    MODE_ARGS=(--load-model-weights "$CKPT" "${CLUST_FLAGS[@]}")
fi
TAG=${TAG:-dpcgrid}
N_EVAL_FILES=${N_EVAL_FILES:-20}
DEBUG_RHO_FIRST=${DEBUG_RHO_FIRST:-1}

# label:delta_min:rho_min   (defaults: the 2x3 grid described above)
POINTS=${POINTS:-"d40r5e2:0.4:0.05 d40r5e3:0.4:0.005 d40r5e4:0.4:0.0005 d10r5e2:0.10:0.05 d10r5e3:0.10:0.005 d10r5e4:0.10:0.0005"}

VAL_DIR=$SCR/data/50k_validation_delana/digitized
mapfile -t ALL < <(ls -1 "$VAL_DIR"/pf_tree_*.parquet)
DATA_FILES=("${ALL[@]:0:$N_EVAL_FILES}")
[[ ${#DATA_FILES[@]} -eq 0 ]] && { echo "no parquets in $VAL_DIR" >&2; exit 2; }
[[ -f "$CKPT" ]] || { echo "no checkpoint at $CKPT" >&2; exit 2; }

mkdir -p "$MODEL_DIR/showers_df_evaluation" "$SCR/wandb"
cd "$REPO" || exit 1
echo "[scan] model  : $MODEL   ($NETCFG)"
echo "[scan] mode   : $MODE"
echo "[scan] ckpt   : $CKPT"
echo "[scan] files  : ${#DATA_FILES[@]} from $VAL_DIR"
echo "[scan] points : $POINTS"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# Knobs a point may set. Reset to the code default before every point so one
# point cannot leak into the next -- the single commonest way a scan lies.
SCAN_KNOBS=(DPC_D_C DPC_RHO_MIN DPC_DELTA_MIN DPC_CORE_R DPC_NMS DPC_DEBUG_RHO
            BAD_TRACK_REMOVAL BAD_TRACK_SIGMA BAD_TRACK_STATS ADD_LONELY_TRACKS
            EVAL_CLS_THRESHOLD EVAL_MASK_THRESHOLD FORCE_TRACK_ASSIGNMENT
            M3D_VALID_ARGMAX M3D_DEBUG_LABELS EC_CKPT_OVERRIDE)

first=1
for pt in $POINTS; do
    LBL=${pt%%:*}; SPEC=${pt#*:}
    for k in "${SCAN_KNOBS[@]}"; do unset "$k"; done
    if [[ "$SPEC" == *"="* ]]; then
        IFS=, read -ra KVS <<< "$SPEC"
        for kv in "${KVS[@]}"; do export "${kv?}"; done
    else                                   # legacy `label:delta:rho`
        IFS=: read -r DMIN RMIN <<< "$SPEC"
        export DPC_DELTA_MIN="$DMIN" DPC_RHO_MIN="$RMIN"
    fi
    [[ -n "${EXTRA_ENV:-}" ]] && for kv in $EXTRA_ENV; do export "${kv?}"; done
    if [[ "$DEBUG_RHO_FIRST" == "1" && $first -eq 1 ]]; then export DPC_DEBUG_RHO=1; fi
    first=0
    # knobs that are CLI flags rather than env vars
    PT_ARGS=()
    [[ -n "${EVAL_CLS_THRESHOLD:-}"  ]] && PT_ARGS+=(--eval-cls-threshold  "$EVAL_CLS_THRESHOLD")
    [[ -n "${EVAL_MASK_THRESHOLD:-}" ]] && PT_ARGS+=(--eval-mask-threshold "$EVAL_MASK_THRESHOLD")
    # model-constructor options are `-o k v`, and eval parses the value with a
    # bare ast.literal_eval, so it must be a Python literal (True, not true).
    [[ -n "${FORCE_TRACK_ASSIGNMENT:-}" ]] && PT_ARGS+=(-o force_track_assignment "$FORCE_TRACK_ASSIGNMENT")
    # Swap the stage-2 EC checkpoint per point. argparse keeps the LAST
    # occurrence of a `store` argument and PT_ARGS is appended after MODE_ARGS,
    # so this overrides the job-level --load-model-weights.
    [[ -n "${EC_CKPT_OVERRIDE:-}" ]] && PT_ARGS+=(--load-model-weights "$EC_CKPT_OVERRIDE")
    OUT="eval_${TAG}_${LBL}.pkl"
    echo "=============================================================="
    echo -n "[scan] point=$LBL  env:"
    for k in "${SCAN_KNOBS[@]}"; do [[ -n "${!k:-}" ]] && echo -n " $k=${!k}"; done
    echo "  -> $OUT"
    echo "[scan] start $(date +%T)"
    $PY -m src.train_lightning1 \
        --delphi --predict \
        --data-test "${DATA_FILES[@]}" \
        --name-output "$OUT" \
        --data-config config_files/config_hits_track_delphi.yaml \
        --network-config "$NETCFG" \
        --model-prefix "$MODEL_DIR" \
        --num-workers 4 --gpus 0 --batch-size "${BATCH_SIZE:-1}" \
        --start-lr 1e-4 --num-epochs 1 --train-batches 1 \
        --fetch-step "${FETCH_STEP:-0.1}" \
        --log-wandb --wandb-displayname "scan_${TAG}_${LBL}" \
        --wandb-projectname mlpf-delphi --wandb-entity optimal-design \
        --tracks \
        "${MODE_ARGS[@]}" ${NETOPTS[@]+"${NETOPTS[@]}"} ${PT_ARGS[@]+"${PT_ARGS[@]}"}
    echo "[scan] point=$LBL exit=$?  end $(date +%T)"
done

echo "[scan] ALL DONE"
ls -la "$MODEL_DIR/showers_df_evaluation/" | grep "$TAG"
