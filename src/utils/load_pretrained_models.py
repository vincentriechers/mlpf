
import torch 

def load_train_model(args, dev, model=None):
    """Warm-start a TRAINING run from --load-model-weights.

    Pass the model `model_setup` already built from --network-config and the
    weights are loaded into it. Without `model` this falls back to the historical
    behaviour, which builds a HARDCODED Gatr_pf_e_noise and ignores
    --network-config entirely.

    That fallback silently trains the wrong architecture for any non-GATr
    warm start: the wrapper from --network-config is discarded, the checkpoint is
    loaded into a GATr model with strict=False so almost no key matches, and what
    trains is a randomly-initialised GATr. It exits 0 and the loss falls
    smoothly. The only outward sign is the loss FAMILY in the logs -- object
    condensation (`loss lv`, `loss beta`) instead of the mask losses the Mask3D /
    Attn-IPA models emit. Caught 2026-09-05 on an Attn-IPA continuation
    (job 45444306, cancelled after 30 min).

    The load is shape-matched and non-strict, and it PRINTS what it did: a warm
    start that matches almost nothing is the failure mode to catch here.
    """
    if model is None:
        from src.models.GATr.Gatr_pf_e_noise import ExampleWrapper as GravnetModel
        return GravnetModel.load_from_checkpoint(
            args.load_model_weights, args=args, dev=0, map_location=dev, strict=False)

    target = getattr(model, "mod", model)
    ckpt = torch.load(args.load_model_weights, map_location="cpu")
    sd = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
    tgt_sd = target.state_dict()
    keep = {k: v for k, v in sd.items() if k in tgt_sd and tgt_sd[k].shape == v.shape}
    missing, unexpected = target.load_state_dict(keep, strict=False)
    print(f"[warm-start] {args.network_config} <- {args.load_model_weights}: "
          f"loaded {len(keep)}/{len(sd)} shape-matching keys "
          f"({len(missing)} missing, {len(unexpected)} unexpected). "
          f"A low ratio here means the checkpoint does not match this architecture.",
          flush=True)
    return model


def load_test_model(args, dev, data_config=None):
    """Build the eval model from the wrapper given in --network-config -- the SAME mechanism
    training uses (model_setup: import_module(args.network_config).get_model(...).mod) -- then
    load the weights into it. This makes the eval model match the --network-config in the
    command instead of a hardcoded class, and it also picks up model flags carried on `args`
    (e.g. --clip-calo-momentum / --unit-calo-direction).

    Weight loading (all shape-matched, strict=False, so mismatched heads/backbones are skipped
    rather than erroring):
      * with --load-model-weights-clustering : graft the EC/PID head from --load-model-weights
        (if given) and then the clustering backbone from --load-model-weights-clustering;
      * otherwise                            : load the full --load-model-weights checkpoint.
    """
    import ast
    from src.utils.import_tools import import_module

    def _state_dict(path):
        c = torch.load(path, map_location="cpu")
        return c["state_dict"] if isinstance(c, dict) and "state_dict" in c else c

    # Build the model corresponding to the wrapper file in the command (like training).
    network_module = import_module(args.network_config, name="_network_module")
    network_options = {k: ast.literal_eval(v) for k, v in args.network_option}
    model, _ = network_module.get_model(data_config, args=args, dev=dev, **network_options)
    model = model.mod

    if getattr(args, "load_model_weights_clustering", None) is not None:
        # Graft the EC/PID head first (throwaway clustering backbone inside it is filtered
        # out by the shape match), then overwrite the clustering backbone.
        if args.load_model_weights is not None:
            ec_sd = _state_dict(args.load_model_weights)
            msd = model.state_dict()
            ec_keep = {k: v for k, v in ec_sd.items() if k in msd and msd[k].shape == v.shape}
            model.load_state_dict(ec_keep, strict=False)
            print(f"[load_test_model] EC/PID head <- {args.load_model_weights}: "
                  f"loaded {len(ec_keep)}/{len(ec_sd)} shape-matching keys")
        m, u = model.load_state_dict(_state_dict(args.load_model_weights_clustering), strict=False)
        print(f"[load_test_model] clustering <- {args.load_model_weights_clustering}: "
              f"{len(m)} missing, {len(u)} unexpected")
    elif args.load_model_weights is not None:
        sd = _state_dict(args.load_model_weights)
        msd = model.state_dict()
        keep = {k: v for k, v in sd.items() if k in msd and msd[k].shape == v.shape}
        m, u = model.load_state_dict(keep, strict=False)
        print(f"[load_test_model] {args.network_config} <- {args.load_model_weights}: "
              f"loaded {len(keep)}/{len(sd)} shape-matching keys ({len(m)} missing, {len(u)} unexpected)")

    return model



def load_test_model2(args, dev):
    if args.load_model_weights is not None and (not args.correction):
            from src.models.gravnet_plus_ecalibration import ExampleWrapper as GravnetModel
            model = GravnetModel.load_from_checkpoint(
                args.load_model_weights, args=args, dev=0, map_location=dev, strict=False
            )

    if args.load_model_weights is not None and args.correction:
            from src.models.gravnet_plus_ecalibration import ExampleWrapper as GravnetModel
            ckpt = torch.load(args.load_model_weights, map_location=dev)

            state_dict = ckpt["state_dict"]

            # Remove PID head weights
            # keys_to_remove = [
            #     k for k in state_dict
            #     if ((k.startswith("ec_model_wrapper_charged.PID_head"))) #or (k.startswith("ec_model_wrapper_neutral.PID_head")) or (k.startswith("ec_model_wrapper_neutral.gatr_pid")))
            # ] #

            # for k in keys_to_remove:
            #     del state_dict[k]
            print("loading state dic clustering")
            model = GravnetModel( args=args, dev=0)
            current_state = model.state_dict()
            filtered_sd = {k: v for k, v in state_dict.items()
                           if k in current_state and current_state[k].shape == v.shape}
            model.load_state_dict(filtered_sd, strict=False)

            # model = GravnetModel.load_from_checkpoint(
            #     args.load_model_weights_clustering, args=args, dev=0, strict=False, map_location=torch.device("cuda:2")
            # )
            import copy
            ckpt2 = torch.load(args.load_model_weights_clustering, map_location=torch.device("cuda:1"))
            sd2 = ckpt2["state_dict"]
            # args2 = copy.copy(args)
            # if "clustering.weight" in sd2:
            #     args2.clustering_space_dim = sd2["clustering.weight"].shape[0]
            model2 = GravnetModel(args=args, dev=0)
            # current_state2 = model2.state_dict()
            # filtered_sd2 = {k: v for k, v in sd2.items()
            #                 if k in current_state2 and current_state2[k].shape == v.shape}
            model2.load_state_dict(sd2, strict=False)
            model2 = model2.to(torch.device("cuda:1"))
            # GravNet backbone has no `gatr` module — copy the full GravNet
            # clustering stack (ScaledGooeyBatchNorm2_1 -> linear1 -> gravnet_blocks
            # -> postprocessing -> clustering/beta, see ExampleWrapper._backbone).
            model.ScaledGooeyBatchNorm2_1 = model2.ScaledGooeyBatchNorm2_1
            model.linear1 = model2.linear1
            model.gravnet_blocks = model2.gravnet_blocks
            model.postprocessing = model2.postprocessing
            model.clustering = model2.clustering
            model.beta = model2.beta
    return model