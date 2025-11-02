
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
