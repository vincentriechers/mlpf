#!/bin/bash
# =============================================================================
# DELPHI clustering evaluation on BSC — HitPF **or** Mask3D/Attn-IPA.
#
# Produces the matched-showers dataframe under
#     <MODEL_DIR>/showers_df_evaluation/<OUTPUT_NAME>
# which is what the DELPHI performance plots and the three-way
# DELANA / HitPF / Mask3D comparison consume.
#
# Runs on the HELD-OUT 50k validation sample (independent seeds), NOT the tail
# of the training folder.
#
#   MODEL=hitpf  sbatch scripts/bsc/eval_delphi_clustering.sh
#   MODEL=mask3d CKPT=/path/to/_epoch=0_step=N.ckpt \
#       sbatch --export=ALL,MODEL=mask3d,CKPT=... scripts/bsc/eval_delphi_clustering.sh
#
# Env: MODEL {hitpf|mask3d}, CKPT, MODEL_DIR, RUN_NAME, N_EVAL_FILES, BATCH_SIZE
#
# TWO TRAPS, both paid for already:
#
#  1. **Pass the SAME `-o` options you trained with.** For Mask3D that means
#     `-o window_size None`; without it the encoder builds flash-attn windowed
#     layers instead of xformers block-diagonal ones, the shapes change, and
#     load_test_model silently keeps only the shape-matching keys (strict=False)
#     — you get a randomly-initialised encoder and a plausible-looking but
#     meaningless evaluation. WATCH THE `loaded N/M shape-matching keys` LINE:
#     it must be 412/412 with 0 missing and 0 unexpected.
#  2. **Eval parses `-o` values with a bare `ast.literal_eval`**
#     (load_pretrained_models.py:33), unlike training which falls back to the raw
#     string (train_utils.py:58). So every `-o` value here must be a valid Python
#     literal: `None`, `3.0`, `True` are fine, a bare word like `focal` raises
#     ValueError. Quote it if you need one.
# =============================================================================
#SBATCH --job-name=delphi_eval_clust
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

export WANDB_MODE=offline
export WANDB_DIR=$SCR
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NUMEXPR_MAX_THREADS=${NUMEXPR_MAX_THREADS:-20}
export NUMEXPR_NUM_THREADS=${NUMEXPR_NUM_THREADS:-20}

MODEL=${MODEL:-hitpf}
case "$MODEL" in
  hitpf)
    NETCFG=src/models/wrapper/example_mode_gatr_noise.py
    DEFAULT_DIR=$SCR/trained-models/hitpf_stage1
    # OC clustering knobs — must match the training recipe
    EXTRA=(--condensation -clust -clust_dim 3 --qmin 3 --use-average-cc-pos 0.98
           --optimizer ranger)
    NETOPTS=()
    ;;
  mask3d)
    NETCFG=src/models/wrapper/example_mode_mask3d.py
    DEFAULT_DIR=$SCR/trained-models/attn_ipa_stage1
    # Mask3D uses Hungarian matching, not object condensation. adamW is the only
    # optimizer attn_ipa/mask3d accept for construction.
    EXTRA=(--optimizer adamW)
    # MUST mirror the training -o options (see trap 1 above). num_queries and
    # use_ipa_decoder change PARAMETER SHAPES, so a mismatch is not cosmetic:
    # load_test_model loads with strict=False and would silently keep only the
    # shape-matching keys, evaluating a half-random model. Defaults here match
    # the plain-Mask3D job; pass USE_IPA_DECODER=True to evaluate an
    # attn_ipa_model checkpoint.
    NETOPTS=(-o window_size None
             -o track_loss_weight "${TRACK_LOSS_WEIGHT:-3.0}"
             -o num_queries "${NUM_QUERIES:-320}"
             -o use_ipa_decoder "${USE_IPA_DECODER:-False}")
    ;;
  *) echo "MODEL must be hitpf or mask3d, got '$MODEL'" >&2; exit 2 ;;
esac

MODEL_DIR=${MODEL_DIR:-$DEFAULT_DIR}
# newest checkpoint unless told otherwise; subshell + `|| true` so head's early
# close does not kill us under pipefail when there are hundreds of ckpts
CKPT=${CKPT:-$( (ls -1t "$MODEL_DIR"/*.ckpt 2>/dev/null || true) | head -n1 )}
RUN_NAME=${RUN_NAME:-eval_${MODEL}}
OUTPUT_NAME=${OUTPUT_NAME:-eval_clustering_delphi_${MODEL}.pkl}

VAL_DIR=$SCR/data/50k_validation_delana/digitized
N_EVAL_FILES=${N_EVAL_FILES:-all}
mapfile -t ALL_FILES < <(ls -1 "$VAL_DIR"/pf_tree_*.parquet 2>/dev/null)
if [[ ${#ALL_FILES[@]} -eq 0 ]]; then echo "no parquets in $VAL_DIR" >&2; exit 2; fi
if [[ "$N_EVAL_FILES" == "all" ]]; then
    DATA_FILES=("${ALL_FILES[@]}")
else
    DATA_FILES=("${ALL_FILES[@]:0:$N_EVAL_FILES}")
fi

if [[ -z "$CKPT" || ! -f "$CKPT" ]]; then
    echo "no checkpoint found (MODEL_DIR=$MODEL_DIR, CKPT=${CKPT:-unset})" >&2; exit 2
fi

mkdir -p "$SCR/wandb" "$SCR/slurm-logs" "$MODEL_DIR/showers_df_evaluation"
cd "$REPO" || exit 1
echo "[eval] model      : $MODEL   ($NETCFG)"
echo "[eval] checkpoint : $CKPT"
echo "[eval] val files  : ${#DATA_FILES[@]} from $VAL_DIR"
echo "[eval] output     : $MODEL_DIR/showers_df_evaluation/$OUTPUT_NAME"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

$PY -m src.train_lightning1 \
    --delphi \
    --predict \
    --data-test "${DATA_FILES[@]}" \
    --name-output "$OUTPUT_NAME" \
    --data-config config_files/config_hits_track_delphi.yaml \
    --network-config "$NETCFG" \
    --model-prefix "$MODEL_DIR" \
    --load-model-weights "$CKPT" \
    --num-workers 4 \
    --gpus 0 \
    --batch-size "${BATCH_SIZE:-1}" \
    --start-lr 1e-4 \
    --num-epochs 1 \
    --train-batches 1 \
    --fetch-step "${FETCH_STEP:-0.1}" \
    --log-wandb \
    --wandb-displayname "$RUN_NAME" \
    --wandb-projectname mlpf-delphi \
    --wandb-entity optimal-design \
    --tracks \
    "${EXTRA[@]}" \
    "${NETOPTS[@]}"

RC=$?
# Capture BEFORE the echo, and exit with it. Without the explicit
# `exit $RC` the script ends on a successful `echo`, so SLURM records
# the job as COMPLETED even when training died — job 45080197 failed
# on its first batch and still showed State=COMPLETED, ExitCode=0:0.
echo "[eval] exit=$RC"
ls -la "$MODEL_DIR/showers_df_evaluation/" 2>/dev/null | tail -5
exit $RC
