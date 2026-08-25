#!/bin/bash
#SBATCH --job-name=gputest
#SBATCH --output=/gpfs/scratch/ehpc1013/vriecher/slurm-logs/gputest-%j.out
#SBATCH --error=/gpfs/scratch/ehpc1013/vriecher/slurm-logs/gputest-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:1
#SBATCH --time=00:05:00
#SBATCH --qos=acc_debug
#SBATCH --account=ehpc1013
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
/gpfs/projects/ehpc1013/vriecher/envs/mlpf-overlay/bin/python - <<'PY'
import torch
print("torch", torch.__version__, "cuda", torch.version.cuda)
print("cuda available:", torch.cuda.is_available(), "| n_gpu:", torch.cuda.device_count())
print("device:", torch.cuda.get_device_name(0), "| capability:", torch.cuda.get_device_capability(0))
a=torch.randn(4096,4096,device='cuda',dtype=torch.bfloat16)
print("matmul bf16 ok:", (a@a).float().sum().isfinite().item())
import xformers.ops as xops
q=torch.randn(2,1024,8,64,device='cuda',dtype=torch.bfloat16)
o=xops.memory_efficient_attention(q,q,q)
print("xformers mem_eff_attention on Hopper OK:", tuple(o.shape))
import dgl, torch_cmspepr, gatr, lightning
print("dgl",dgl.__version__,"| gatr",gatr.__version__,"| lightning",lightning.__version__)
PY
