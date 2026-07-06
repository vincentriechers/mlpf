#!/usr/bin/env python

import os
import sys
import glob
import torch
import lightning as L

torch.set_float32_matmul_precision("high")  # TF32 tensor cores for FP32 GA matmuls (Blackwell sm_120)
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.strategies import DDPStrategy

sys.path.append(os.path.join(os.path.dirname(__file__), "../"))

from src.utils.parser_args import parser
from src.utils.train_utils import (
    train_load,
    test_load,
    get_samples_steps_per_epoch,
    model_setup,
    set_gpus,
)
from src.utils.load_pretrained_models import (
    load_train_model,
    load_test_model,
    load_test_model2
)
from src.utils.callbacks import (
    get_callbacks,
    get_callbacks_eval,
)


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------

def setup_wandb(args):
    # save_dir MUST be off AFS: WandbLogger defaults save_dir="." (repo cwd on AFS, quota-limited)
    # and passes it to wandb.init as `dir`, overriding the WANDB_DIR env. Writing offline run data
    # to a near-full AFS volume can block rank-0 writes and re-trigger the DDP collective-timeout hang.
    return WandbLogger(
        project=args.wandb_projectname,
        entity=args.wandb_entity,
        name=args.wandb_displayname,
        save_dir=os.environ.get("WANDB_DIR", "/data/mgarciam/wandb"),
    )


def build_trainer(args, gpus, logger, training=True):
    callbacks = get_callbacks(args) if training else get_callbacks_eval(args)
    if args.correction and training:
        strategy = DDPStrategy(find_unused_parameters=True)
    elif training:
        strategy = DDPStrategy(static_graph=True)
    else:
        strategy = "auto"
    #DDPStrategy(find_unused_parameters=True) #
    return L.Trainer(
        gradient_clip_val=1.0, gradient_clip_algorithm="norm",
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
        precision="bf16-mixed" if training else "32",
    )


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def main():
    args = parser.parse_args()
    # torch.autograd.set_detect_anomaly(True)  # debug only — very slow in production

    training_mode = not args.predict
    # GLOBAL rank across all nodes (SLURM multi-node aware). Used for per-rank file
    # sharding in train_load(); must be the global rank so every GPU on every node
    # gets a DISJOINT shard. SLURM_PROCID is set by srun on multi-node; falls back to
    # torchrun's RANK, then the single-node Lightning launcher's LOCAL_RANK.
    args.local_rank = int(
        os.environ.get("SLURM_PROCID", os.environ.get("RANK", os.environ.get("LOCAL_RANK", 0)))
    )
    args.is_muons = True

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
        model = load_train_model(args, dev)

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

        trainer.fit(
            model=model,
            train_dataloaders=train_loader,
            val_dataloaders=val_loader,
            ckpt_path=args.resume_ckpt,
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
            model = load_test_model(args, dev)

        trainer = build_trainer(args, gpus, wandb_logger, training=False)

        for name, get_test_loader in test_loaders.items():
            test_loader = get_test_loader()
            trainer.validate(
                model=model,
                dataloaders=test_loader,
            )


if __name__ == "__main__":
    main()
