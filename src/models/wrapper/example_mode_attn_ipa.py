import torch

from src.models.Mask3D.attn_ipa_model import ExampleWrapper as AttnIPAModel


class GraphTransformerNetWrapper(torch.nn.Module):
    def __init__(self, args, dev, **kwargs):
        super().__init__()
        self.mod = AttnIPAModel(args, dev, **kwargs)

    def forward(self, g, step_count):
        return self.mod(g, step_count)


def get_model(data_config, args, dev, **kwargs):
    print("Attn-IPA model options:", kwargs)
    model = GraphTransformerNetWrapper(args, dev, **kwargs)
    model_info = {}
    return model, model_info


def get_loss(data_config, **kwargs):
    return torch.nn.MSELoss()
