#!/usr/bin/env python

import os
import sys
import glob
import torch
torch.set_float32_matmul_precision("high")
import lightning as L
from lightning.pytorch.callbacks import Callback as _LCallback
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.strategies import DDPStrategy

sys.path.append(os.path.join(os.path.dirname(__file__), "../"))

from src.utils.parser_args import parser
from src.utils.train_utils import (
    train_load,
    test_load,
    get_samples_steps_per_epoch,
    get_global_rank_and_world_size,
    model_setup,
    set_gpus,
)
from src.utils.load_pretrained_models import (
    load_train_model,
    load_test_model,
)
from src.utils.callbacks import (
    get_callbacks,
    get_callbacks_eval,
)


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------

def setup_wandb(args):
    # log_model is off by default — staging checkpoints into wandb's cache
    # blew the home quota on BSC. The local checkpoints under --model-prefix
    # are unaffected. Re-enable with WANDB_LOG_MODEL=all if needed.
    log_model_env = os.environ.get("WANDB_LOG_MODEL", "false").lower()
    log_model = "all" if log_model_env in ("all", "true", "1") else False
    # save_dir MUST be off AFS: WandbLogger defaults save_dir="." (repo cwd on AFS, quota-limited)
    # and passes it to wandb.init as `dir`, overriding the WANDB_DIR env. Writing offline run data
    # to a near-full AFS volume can block rank-0 writes and re-trigger the DDP collective-timeout hang.
    return WandbLogger(
        project=args.wandb_projectname,
        entity=args.wandb_entity,
        name=args.wandb_displayname,
        log_model=log_model,
        save_dir=os.environ.get("WANDB_DIR", "/data/mgarciam/wandb"),
    )


class _UnusedParamsDiag(_LCallback):
    # MASK3D_DIAG_UNUSED=1 — print parameter names whose grad is None after
    # the first training backward (rank 0 only), then no-op. Used to pin down
    # which submodule is silently disconnected from the loss graph when DDP
    # raises "parameters were not used in producing the loss".
    def __init__(self):
        self.fired = False

    def on_after_backward(self, trainer, pl_module):
        if self.fired or not trainer.is_global_zero:
            return
        self.fired = True
        unused = sorted(
            n for n, p in pl_module.named_parameters()
            if p.requires_grad and p.grad is None
        )
        sep = "=" * 72
        lines = [sep, "MASK3D_DIAG_UNUSED — first backward (rank 0)", sep]
        if unused:
            lines.append(f"count: {len(unused)}")
            lines.extend(f"  {n}" for n in unused)
        else:
            lines.append("(none — all parameters are in the autograd graph)")
        lines.append(sep)
        print("\n".join(lines), file=sys.stderr, flush=True)


def build_trainer(args, gpus, logger, training=True):
    callbacks = get_callbacks(args) if training else get_callbacks_eval(args)
    # Plain DDP works for the default model config (`share_decoder_heads=True`,
    # `per_subsystem_input=False`). If either is flipped — or any other
    # config that leaves some params unused on some steps — DDP errors out;
    # we set `find_unused_parameters=True` whenever the network options
    # request it. ~5–15% slower but correct-by-construction.
    netopt = dict(getattr(args, "network_option", []) or [])
    diag_unused = os.environ.get("MASK3D_DIAG_UNUSED", "") == "1"
    needs_find_unused = (
        netopt.get("per_subsystem_input", "False") == "True"
        or netopt.get("per_subsystem_loss", "False") == "True"
        or netopt.get("share_decoder_heads", "True") == "False"
        or netopt.get("num_subsystems", "1") not in ("1", "")
        or diag_unused
    )
    if args.correction and training:
        # energy-correction training leaves some params unused per step (origin/main)
        strategy = DDPStrategy(find_unused_parameters=True)
    elif args.correction:
        strategy = "auto"
    elif training:
        strategy = (
            DDPStrategy(find_unused_parameters=True) if needs_find_unused else "ddp"
        )
    else:
        # Eval / --predict without --correction: let Lightning pick the
        # strategy (SingleDeviceStrategy for one GPU). Newer Lightning
        # rejects `strategy=None`; the --correction path already uses
        # "auto" for the same reason.
        strategy = "auto"

    if training and diag_unused:
        callbacks = list(callbacks) + [_UnusedParamsDiag()]

    # Gradient clipping is REQUIRED for stability of this stack (we observed
    # NaN/Inf cost crashes in scipy.linear_sum_assignment around step 36k of
    # a no-clip run on H100 / bf16 — bidirectional CA + windowed attention
    # can spike logits before the matcher sees them). 0.1 matches hepattn's
    # CLD config; the trainer applies it to the global gradient L2 norm.
    grad_clip_val = float(getattr(args, "gradient_clip_val", 0.1))

    # Mixed precision: --use-amp turns on bf16-mixed (matches hepattn). bf16
    # is the right choice on H100/A100 — wider exponent range than fp16, no
    # loss-scaling needed, and stable with our LayerNorm-heavy backbone.
    precision = "bf16-mixed" if getattr(args, "use_amp", False) else "32-true"

    return L.Trainer(
        callbacks=callbacks,
        accelerator="gpu",
        devices=gpus,
        num_nodes=getattr(args, "num_nodes", 1),
        default_root_dir=args.model_prefix,
        logger=logger,
        max_epochs=args.num_epochs if training else None,
        strategy=strategy,
        limit_train_batches=args.train_batches if training else None,
        limit_val_batches=5 if training else None,
        gradient_clip_val=grad_clip_val if training else None,
        gradient_clip_algorithm="norm",
        precision=precision,
    )


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def main():
    args = parser.parse_args()
    # torch.autograd.set_detect_anomaly(True)  # debug only — very slow in production

    # Opt-in determinism. Default None = unseeded, i.e. exactly the previous
    # behaviour. Without this an A/B comparison (e.g. --pid-class-weighting on
    # vs off) differs by RNG as well as by the thing under test — weight init,
    # per-rank file shuffling in to_filelist, and the DataIter seeds all vary —
    # so a small difference between two arms cannot be attributed to the change.
    # `workers=True` also seeds the DataLoader workers, which is where the file
    # shuffle actually happens.
    if getattr(args, "seed", None) is not None:
        L.seed_everything(int(args.seed), workers=True)
        print(f"[seed] seed_everything({args.seed}, workers=True)")

    training_mode = not args.predict
    # GLOBAL rank across all nodes. Used for per-rank file sharding in
    # train_load() and for the DataIter names in the log, so it must be the
    # global rank or every GPU trains on the same shard. Launcher detection
    # lives in get_global_rank_and_world_size() — rank and world size have to be
    # read from the SAME launcher, and reading SLURM_PROCID first (as this did)
    # is wrong under torchrun-inside-srun, which is what the BSC scripts use.
    args.local_rank, _ = get_global_rank_and_world_size()
    args.is_muons = True
    if args.delphi:
        # DELPHI pf_trees have a 14-feature CALO-only X_hit (X_hit[:,14] of the ILD
        # path would crash) and no pandora branches; positions are handled in the
        # dataset code via the same flag (cm -> mm).
        args.ILD = False
        args.pandora = False
        assert not args.allegro, "--delphi and --allegro are mutually exclusive"

    # --------------------------------------------------
    # Data
    # --------------------------------------------------
    args = get_samples_steps_per_epoch(args)

    if training_mode:
        files = []
        for folder in args.data_train:
            files.extend(glob.glob(folder + "*.parquet"))
        args.data_train = files
        train_loader, val_loader, data_config, train_input_names = train_load(args)
    else:
        test_loaders, data_config = test_load(args)

    # --------------------------------------------------
    # Model & devices
    # --------------------------------------------------
    model = model_setup(args, data_config)
    gpus, dev = set_gpus(args)

    if training_mode and args.load_model_weights:
        # pass the model built from --network-config; without it load_train_model
        # substitutes a hardcoded GATr and the warm start trains the wrong model.
        model = load_train_model(args, dev, model=model)

    # --------------------------------------------------
    # Logger
    # --------------------------------------------------
    wandb_logger = setup_wandb(args)

    # --------------------------------------------------
    # Training
    # --------------------------------------------------
    if training_mode:
        trainer = build_trainer(args, gpus, wandb_logger, training=True)
        args.local_rank = trainer.global_rank

        # --resume-ckpt restores weights + optimizer + LR scheduler + global
        # step from a Lightning checkpoint. Default None = fresh run.
        resume_ckpt = getattr(args, "resume_ckpt", None) or None
        trainer.fit(
            model=model,
            train_dataloaders=train_loader,
            val_dataloaders=val_loader,
            ckpt_path=resume_ckpt,
        )

    # --------------------------------------------------
    # Evaluation
    # --------------------------------------------------
    if args.data_test:
        # Network-config-aware weight loading for the CLD/SO(2) ref-node variant. Its forward
        # (per-event ẑ ref-node splicing) is NOT reproduced by the hardcoded standard
        # Gatr_pf_e_noise that load_test_model builds, so for that architecture we load the
        # weights into the model that model_setup already built from --network-config.
        # Standard models (no _splice_cld_ref_nodes) keep the original load_test_model path,
        # so nothing changes for them.
        # Full-eval grafting (load EC/PID head from --load-model-weights + clustering backbone
        # from --load-model-weights-clustering into the --network-config model) is generic
        # (shape-matched key loading); it works for any clustering model built from the
        # network-config, not only the CLD-ref-node one. Gate it on having a clustering ckpt.
        _netcfg_aware = bool(getattr(args, "load_model_weights_clustering", None))

        def _state_dict(path):
            _c = torch.load(path, map_location="cpu")
            return _c["state_dict"] if isinstance(_c, dict) and "state_dict" in _c else _c

        if _netcfg_aware and args.correction and args.load_model_weights:
            # Full eval: graft the EC/PID head from --load-model-weights onto the SO(2)
            # clustering from --load-model-weights-clustering. Load the EC checkpoint first
            # (populates the EC head + its own standard clustering backbone), then overwrite
            # the clustering backbone with the SO(2) clustering checkpoint.
            print("[eval] CLD full eval: EC/PID head <-", args.load_model_weights)
            # The EC checkpoint may carry a DIFFERENT clustering backbone (e.g. in_s=2,
            # point-only clustering Linear(3,3)) than this CLD model — those keys are
            # thrown away and replaced by the SO(2) clustering checkpoint below. But
            # load_state_dict(strict=False) does NOT skip SIZE mismatches (only missing/
            # unexpected keys), so filter to shape-matching keys and load only those (the
            # EC/PID head + anything compatible).
            _ec_sd = _state_dict(args.load_model_weights)
            _msd = model.state_dict()
            _ec_keep = {k: v for k, v in _ec_sd.items() if k in _msd and _msd[k].shape == v.shape}
            _ec_skip = [k for k in _ec_sd if k not in _ec_keep]
            _ec_skip_head = [k for k in _ec_skip if "energy_correction" in k or "ec_model" in k]
            model.load_state_dict(_ec_keep, strict=False)
            print(
                f"[eval]   EC ckpt: loaded {len(_ec_keep)} shape-matching keys, "
                f"skipped {len(_ec_skip)} (throwaway backbone); EC-head keys skipped: "
                f"{len(_ec_skip_head)} (should be 0)"
            )
            print("[eval]              clustering  <-", args.load_model_weights_clustering)
            _m2, _u2 = model.load_state_dict(_state_dict(args.load_model_weights_clustering), strict=False)
            print(f"[eval]   clustering ckpt: {len(_m2)} missing, {len(_u2)} unexpected")
        elif _netcfg_aware:
            # Clustering-only eval into the --network-config model.
            print("[eval] CLD clustering-only eval: clustering <-", args.load_model_weights_clustering)
            _m, _u = model.load_state_dict(_state_dict(args.load_model_weights_clustering), strict=False)
            print(f"[eval]   loaded state_dict: {len(_m)} missing, {len(_u)} unexpected keys")
        elif args.load_model_weights:
            model = load_test_model(args, dev, data_config)

        trainer = build_trainer(args, gpus, wandb_logger, training=False)

        for name, get_test_loader in test_loaders.items():
            test_loader = get_test_loader()
            trainer.validate(
                model=model,
                dataloaders=test_loader,
            )


if __name__ == "__main__":
    main()
