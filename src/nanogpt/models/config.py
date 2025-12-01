
from dataclasses import dataclass
from torch import nn

@dataclass
class ModuleConfig:
    vocabulary_size: int
    batch_size: int
    block_size: int
    device: str


class ConfigurableModule(nn.Module):
    def __init__(self, config: ModuleConfig):
        super().__init__()

        self.config = config
        self.batch_size = config.batch_size
        self.block_size = config.block_size
        self.device = config.device

    def grouped_parameters(
        self,
        learning_rate: float = 1e-5,
        weight_decay: float = 0.01,
        no_weight_decay_list: list = []
    ):
        """
        Separate parameters into two groups:
        those that should experience weight decay and those that won't.
        """
        grouped_params = []
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue  # frozen weights
            if param.ndim < 2 or 'bias' in name or 'LayerNorm' in name or name in no_weight_decay_list:
                grouped_params.append({
                    "name": name,
                    "params": [param],
                    "lr": learning_rate,
                    "weight_decay": 0.0
                })
            else:
                grouped_params.append({
                    "name": name,
                    "params": [param],
                    "lr": learning_rate,
                    "weight_decay": weight_decay
                })
        return grouped_params