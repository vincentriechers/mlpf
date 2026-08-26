"""Network config for EVALUATING a Mask3D / Attn-IPA checkpoint.

Training uses `example_mode_attn_ipa.py` -> `attn_ipa_model.AttnIPAModel`, which
is a lean training module: forward + loss + wandb logging, and **no evaluation
path at all** (`on_validation_epoch_end` is `pass`, it never writes showers). So
a checkpoint trained with it cannot be turned into the matched-showers dataframe
the DELPHI evaluation needs.

`mask3d_model.ExampleWrapper` is the superset that owns that machinery —
`create_and_store_graph_output_mask3d`, the EC adapter, `--predict`, ranger/lion
— and with `use_ipa_decoder=True` it builds the *same* architecture
(`AttnIPABackbone` + `IPADecoder`). That compatibility is deliberate:
`mask3d_model.py:394` binds the backbone to `self.encoder` rather than
`self.input_net` specifically so `attn_ipa_model` checkpoint keys line up.

So the chain is:  train with example_mode_attn_ipa  ->  evaluate with this file,
loading the same .ckpt via --load-model-weights.

**Pass the SAME `-o` options you trained with.** Every architecture-affecting
default is already identical between the two classes (verified by diffing both
constructors), so the only thing that can break checkpoint loading is a `-o`
override used at training and forgotten at eval — e.g. our DELPHI runs use
`-o window_size None`, without which the encoder builds flash-attn windowed
layers instead of the xformers block-diagonal ones.

Three *loss-only* defaults do differ between the classes, and are pinned below to
the `attn_ipa_model` values so a validation loss computed here means the same
thing as the training loss:

    aux_layer_weight   attn_ipa 1.0    mask3d 0.0
    null_weight        attn_ipa 0.25   mask3d 1.0
    track_loss_weight  attn_ipa 1.0    mask3d 3.0

They are `setdefault`s, so an explicit `-o` still wins.
"""
import torch

from src.models.Mask3D.mask3d_model import ExampleWrapper as Mask3DModel


# Defaults that differ between attn_ipa_model.AttnIPAModel and
# mask3d_model.ExampleWrapper. Loss-only: they do not change any parameter
# shape, so they never affect whether a checkpoint loads — only what the
# reported loss means.
_ATTN_IPA_LOSS_DEFAULTS = {
    "aux_layer_weight": 1.0,
    "null_weight": 0.25,
    "track_loss_weight": 1.0,
}


class GraphTransformerNetWrapper(torch.nn.Module):
    def __init__(self, args, dev, **kwargs):
        super().__init__()
        # IPA decoder + AttnIPABackbone == the attn_ipa architecture. Left as a
        # setdefault rather than forced, so `-o use_ipa_decoder False` still
        # gives the plain Mask3D (MaskFormerDecoder) variant from this config.
        kwargs.setdefault("use_ipa_decoder", True)
        for k, v in _ATTN_IPA_LOSS_DEFAULTS.items():
            kwargs.setdefault(k, v)
        self.mod = Mask3DModel(args, dev, **kwargs)

    def forward(self, g, step_count):
        return self.mod(g, step_count)


def get_model(data_config, args, dev, **kwargs):
    print("Mask3D (eval) model options:", kwargs)
    model = GraphTransformerNetWrapper(args, dev, **kwargs)
    model_info = {}
    return model, model_info


def get_loss(data_config, **kwargs):
    return torch.nn.MSELoss()
