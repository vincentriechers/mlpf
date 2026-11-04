# DELPHI particle flow (branch `delphi-mask3d`)

Hit-level particle flow for the DELPHI detector at LEP: trained on DELPHI simulation of Z -> qq at 91 GeV, applied to
recorded 1994 data, and compared with the standard DELPHI reconstruction (DELANA). The input files (`pf_tree_<seed>.parquet`)
come from the `delphi_converter` repository (github.com/vincentriechers/delphi_converter); their schema is in that
repository's `conversion/docs/PARQUET_FORMAT.md`.

## What this branch adds to MLPF

- **`--delphi` input path**: positions cm -> mm, DELPHI hit types, data config `config_files/config_hits_track_delphi.yaml`.
- **Stage 1, two clustering models** on the same input:
  - *HitPF*: GATr + object condensation + density-peak clustering (`src/models/wrapper/example_mode_gatr_noise.py`);
  - *Mask3D*: encoder + MaskFormer decoder with Hungarian matching (`src/models/Mask3D/mask3d_model.py`, wrapper
    `src/models/wrapper/example_mode_mask3d.py`). The Attn-IPA variant (`example_mode_attn_ipa.py`) is evaluated through
    the same wrapper with `-o use_ipa_decoder True`.
- **Stage 2**, energy correction + PID (`src/models/energy_correction_NN_v1.py`): trained on truth clusters, so it does
  not depend on stage 1. The heads read only the cluster's cells and tracks, so one stage-2 checkpoint is grafted onto
  either model at evaluation. Optional **noise class** (`EC_NOISE_CLUSTERS`): groups of cells without a particle become a
  third neutral class, and clusters classified as noise are dropped at inference.
- **Post-hoc cluster repair** (`src/layers/postprocess_labels.py`), **inference on recorded data** without truth
  (`PREDICT_NO_TRUTH`), a per-hit **embedding dump** (`DUMP_EMBEDDINGS`).
- **Tests** in `tests/`, plain scripts run from the repository root: `python3 tests/test_postprocess_labels.py` etc. They
  need torch (and scipy for `test_noise_clusters.py`), not dgl; `test_compact_labels.py` imports `inference_oc` and so
  needs the full training environment.

## Documentation outside this repository

Kept on CERN EOS, not on GitHub:

- `/eos/home-v/vriecher/delphi_study/bsc_training_howto.md`: BSC access, environment, data transfer, every script below,
  measured throughput and the known traps. **Read it before the first job.**
- `/eos/home-v/vriecher/delphi_study/README.md`: the study workspace (data sets, write-ups); the current status is in
  `status_2026-11-03/README.md` next to it.
- `/eos/home-v/vriecher/tmp_retrain/launch_*.sh` and `NOTES.md`: the exact launch commands and job ids of each
  training and evaluation round.

## Scripts in this folder (MareNostrum 5, SLURM)

They hardcode the project's BSC paths (`PROJ`, `SCR`, `REPO`, the `#SBATCH` output/error lines and the account) and the
Python of the BSC environment; adapt them to your own area. Submit from the repository root.

| script | what it runs |
|---|---|
| `train_delphi_hitpf_4gpu.sh` | HitPF stage 1, 4 GPUs |
| `train_delphi_mask3d.sh` | Mask3D stage 1, 4 GPUs |
| `train_delphi_stage2.sh` | stage 2 (energy correction + PID, optional noise class) |
| `scan_dpc_grid.sh` | evaluation with post-hoc settings, several settings per job; `MODE=full` grafts stage 2 on |
| `eval_delphi_clustering.sh`, `eval_delphi_full.sh` | single evaluations: stage 1 alone, stage 1 + stage 2 |
| `smoke_delphi_hitpf.sh`, `smoke_delphi_attn_ipa.sh` | 1-GPU smoke tests (the Attn-IPA one is also its production job) |
| `launch_mask3d_scan.sh`, `eval_mask3d_scan.sh` | the Mask3D variant scan of September 2026 |
| `gputest_bsc.sh`, `gpu_util_summary.sh`, `read_offline_wandb.py` | environment check, GPU-utilisation summary, offline W&B reader |
| `diag_attn_ipa_nan.py`, `diag_track_cost_reach.py` | one-batch diagnostics |

## Training (current recipe, 4 x H100)

`D` is a directory of `pf_tree_*.parquet`. `TRAIN_BATCHES` is the number of steps per epoch per GPU; the values below
are for the 4 933 training files of 100 events (how to choose them for another set: the BSC how-to, *Planning the long
runs*).

```bash
D=<training directory>
RUN_NAME=hitpf_s1 DATA_TRAIN=$D TRAIN_BATCHES=6043 NUM_EPOCHS=17 EXTRA_ARGS="--no-track-noise-rule --seed 42" \
  sbatch --export=ALL --qos=acc_ehpc --time=7:00:00 scripts/bsc/train_delphi_hitpf_4gpu.sh
RUN_NAME=m3d_s1 DATA_TRAIN=$D NUM_EPOCHS=32 TRAIN_BATCHES=1888 BATCH_SIZE=64 SEED=42 NUM_QUERIES=320 \
  TRACK_LOSS_WEIGHT=3.0 NETWORK_OPTS=--no-track-noise-rule \
  sbatch --export=ALL --qos=acc_ehpc --time=8:00:00 scripts/bsc/train_delphi_mask3d.sh
# stage 2, shared by both models; leave out the two EC_NOISE_* variables for the version without the noise class
EC_NOISE_CLUSTERS=12 EC_NOISE_EMIN=0 EC_NEUTRAL_INDIST=0.95 PID_WEIGHTING_NEUTRAL=sqrt_inv RUN_NAME=s2 DATA_TRAIN=$D \
  NUM_EPOCHS=2 TRAIN_BATCHES=6043 SEED=42 NETWORK_OPTS=--no-track-noise-rule \
  sbatch --export=ALL --qos=acc_ehpc --time=13:00:00 scripts/bsc/train_delphi_stage2.sh
```

On every multi-GPU run, check the `[shard]` lines in the log: distinct ranks, `world_size 4`, a quarter of the files each.

## Evaluation on a set of pf_tree parquets

`scan_dpc_grid.sh` with `MODE=full` runs stage 1 + stage 2 on `N_EVAL_FILES` files of `VAL_DIR` after skipping
`SKIP_FILES`, once per point of `POINTS` (`label:KEY=VAL,KEY=VAL,...`). The points below are the final inference settings.

```bash
HP="hp:DPC_DELTA_MIN=0.10,DPC_RHO_MIN=0.02,BAD_TRACK_REMOVAL=0,CLUSTER_POST=1,POST_TRACK_LONELY=1,POST_TRACK_LONELY_PMAX=45.6,POST_TRACK_PMAX_ALL=45.6,POST_TRACK_ONE_PER_CLUSTER=1"
M3D="m3d:EVAL_CLS_THRESHOLD=0.5,M3D_VALID_ARGMAX=1,FORCE_TRACK_ASSIGNMENT=False,CLUSTER_POST=1,POST_TRACK_ATTACH_R=200,POST_TRACK_ATTACH_ANY=1,POST_TRACK_LONELY=1,POST_TRACK_LONELY_PMAX=45.6,BAD_TRACK_REMOVAL=0,POST_TRACK_PMAX_ALL=45.6,EVAL_TRACK_MASK_THRESHOLD=0.02,POST_TRACK_ONE_PER_CLUSTER=1"
MODE=full MODEL=hitpf  TAG=test VAL_DIR=<dir> SKIP_FILES=0 N_EVAL_FILES=20 CKPT=<stage-1 ckpt> EC_CKPT=<stage-2 ckpt> \
  MODEL_DIR=<out> POINTS="$HP" sbatch --export=ALL scripts/bsc/scan_dpc_grid.sh
MODE=full MODEL=mask3d TAG=test VAL_DIR=<dir> SKIP_FILES=0 N_EVAL_FILES=20 CKPT=<stage-1 ckpt> EC_CKPT=<stage-2 ckpt> \
  MODEL_DIR=<out> NUM_QUERIES=320 POINTS="$M3D" sbatch --export=ALL scripts/bsc/scan_dpc_grid.sh
# -> <out>/showers_df_evaluation/eval_test_hp.pkl0_0_None.pt and eval_test_m3d.pkl0_0_None.pt
```

- A stage 2 trained with the noise class must be evaluated with it: add `EXTRA_ENV="EC_NOISE_CLUSTERS=12 EC_NOISE_DROP=0.5"`.
- Recorded data (no generator particles): add `PREDICT_NO_TRUTH=1` to `EXTRA_ENV`. The output then has one row per predicted
  cluster and no truth columns.
- Check the two `loaded N/M ... keys` lines in the log: loading is `strict=False`, so a mismatch is silent.
- Event order in the output is not sorted-file order (`--num-workers 4` interleaves four files).

Plots (any machine with torch, pandas, matplotlib, awkward and pyarrow; no GPU):

```bash
MLPF_PLOT_USETEX=0 python3 -m src.evaluation.full_evaluation_delphi_models \
  --model HitPF='<out>/showers_df_evaluation/eval_test_hp.pkl*' --model Mask3D='<out>/showers_df_evaluation/eval_test_m3d.pkl*' \
  --val-dir <dir> --delana --skip-files 0 --n-files 20 --output-dir <plot dir>
```

Efficiency, fake rate, energy resolution and PID per particle, and event energy and mass with DELANA overlaid
(`--help` for the options). `--skip-files` / `--n-files` must be the slice the models were evaluated on; the script
prints `MISMATCH` when the truth medians disagree. `MLPF_PLOT_USETEX=0`: where LaTeX is installed the default (`auto`)
switches it on, and the n_hits diagnostic's "≥" label then fails to compile. The event-level comparison with DELANA used in the study (visible energy, data against
simulation) is in the `delphi_converter` repository, `analysis/`.

## Environment switches

All unset by default, which reproduces the earlier behaviour. `scan_dpc_grid.sh` resets every knob of a point before the next.

| switch | where | effect |
|---|---|---|
| `EC_NOISE_CLUSTERS=<r>` | stage 2 | noise class: cells without a particle linked within `r` (units of `pos_hits_xyz`, mm with `--delphi`) form noise pseudo-clusters; needed at training and at evaluation |
| `EC_NOISE_EMIN`, `EC_NOISE_DROP` | stage 2 | energy floor of a noise cluster [GeV]; noise probability above which a neutral cluster is dropped (0.5) |
| `EC_NEUTRAL_INDIST` | stage 2 training | neutral-head cut on abs(E_true - E_reco)/E_true (0.6; current recipe 0.95) |
| `PREDICT_NO_TRUTH=1` | inference | recorded data: placeholder truth, matching off, one row per predicted cluster |
| `CLUSTER_POST=1` | inference | master switch of the post-hoc repair; the `POST_*` knobs act only with it |
| `POST_TRACK_ATTACH_R`, `POST_TRACK_ATTACH_ANY` | inference | attach unassigned tracks to the nearest cluster within R [mm]; also to clusters that already hold a track |
| `POST_TRACK_LONELY`, `POST_TRACK_LONELY_PMAX`, `_MAXD0`, `_MAXZ0` | inference | remaining tracks become charged particles; momentum [GeV] and impact-parameter [cm] cuts |
| `POST_TRACK_PMAX_ALL`, `POST_TRACK_ONE_PER_CLUSTER` | inference | ignore tracks above p [GeV]; keep one track per cluster |
| `POST_NEUTRAL_MINCELLS`, `_E` | inference | dissolve trackless barrel EM clusters above E [GeV] with fewer cells |
| `POST_RECHECK`, `POST_STATS` | inference | re-run the track/calorimeter check after attachment (undoes it on DELPHI); print counts |
| `DPC_D_C`, `DPC_RHO_MIN`, `DPC_DELTA_MIN`, `DPC_CORE_R`, `DPC_NMS` | HitPF inference | density-peak clustering parameters |
| `BAD_TRACK_REMOVAL`, `BAD_TRACK_SIGMA`, `BAD_TRACK_STATS` | inference | track/calorimeter energy consistency check (on by default; final settings 0) |
| `M3D_VALID_ARGMAX=1` | Mask3D inference | only queries above the class threshold compete for a hit |
| `M3D_TRACK_PIN=1` | Mask3D inference | a track joins the query it seeded (meaningful only for a model trained with `-o track_pin True`) |
| `EVAL_CLS_THRESHOLD`, `EVAL_MASK_THRESHOLD`, `EVAL_TRACK_MASK_THRESHOLD`, `FORCE_TRACK_ASSIGNMENT`, `EC_CKPT_OVERRIDE`, `CLUST_CKPT_OVERRIDE` | `scan_dpc_grid.sh` | turned into the matching command-line options per point |
| `ADD_LONELY_TRACKS`, `LONELY_TRACK_PMAX` | inference | older lone-track promotion in the stage-2 matching path |
| `DUMP_EMBEDDINGS=<dir>`, `DUMP_EMBEDDINGS_EVERY` | inference | per-hit embeddings and labels as npz |
| `DUMP_GRAPHS`, `DUMP_NH_FATE`, `M3D_DEBUG_LABELS`, `DPC_DEBUG_RHO`, `MASK3D_DIAG_UNUSED` | diagnostics | graph dumps, neutral-hadron energy trace, label and density printouts, DDP unused-parameter detection |

Training flags specific to DELPHI: `--delphi`, `--no-track-noise-rule` (skip the CLD rule that relabels tracks as noise),
`--seed`, `--beta-noise-weight`, `--beta-noise-weight-track`, `--pid-class-weighting[-charged|-neutral]`.
