#!/bin/bash
# Summarise a GPU-utilisation CSV written by the sbatch scripts' sampler.
#   usage: bash scripts/bsc/gpu_util_summary.sh <job>.gpu.csv [skip_seconds]
#
# CSV columns (nvidia-smi --format=csv,noheader,nounits):  index, util.gpu, memory.used(MiB)
#
# `skip_seconds` drops the first N samples per GPU, which are job startup (model
# build, first data fetch) and would drag the mean down; default 45, matching the
# ~40 s startup these jobs show before step 10.
#
# What to look for:
#   mean util   high (>85 %) = compute-bound, the GPU is the bottleneck: good.
#               low  (<60 %) = starved, almost always the data loader here.
#   %% idle      fraction of samples at 0 % util — this is the smoking gun for
#               loader stalls; it was ~45 % before NUM_WORKERS/PREFETCH_FACTOR
#               were raised.
#   peak mem    against 65 536 MiB tells you the batch-size headroom.
set -uo pipefail
CSV=${1:?usage: gpu_util_summary.sh <csv> [skip_seconds]}
SKIP=${2:-45}
[[ -r "$CSV" ]] || { echo "cannot read $CSV" >&2; exit 2; }

awk -F', *' -v skip="$SKIP" '
    { n[$1]++; if (n[$1] <= skip) next
      c[$1]++; u[$1]+=$2; if ($2+0==0) z[$1]++
      if ($2+0 > umax[$1]) umax[$1]=$2
      if ($3+0 > m[$1])    m[$1]=$3 }
    END {
        printf "%-5s %9s %9s %9s %11s %9s\n", "gpu","samples","meanUtil","maxUtil","idle@0%","peakMiB"
        # plain index walk rather than gawk-only asorti(), so this also runs
        # under mawk/busybox awk
        for (i = 0; i < 16; i++) {
            g = i "";  if (!(g in c)) continue
            printf "%-5s %9d %8.1f%% %8d%% %10.1f%% %9d\n",
                   g, c[g], u[g]/c[g], umax[g], 100*z[g]/c[g], m[g]
            tc+=c[g]; tu+=u[g]; tz+=z[g]; if (m[g]>tm) tm=m[g]
        }
        if (tc) printf "%-5s %9d %8.1f%% %9s %10.1f%% %9d\n", "ALL", tc, tu/tc, "-", 100*tz/tc, tm
    }' "$CSV"
