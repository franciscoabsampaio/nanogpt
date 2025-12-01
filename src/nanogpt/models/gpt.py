from .transformer import Config, Transformer


class GPT2(Transformer):
    def __init__(
        self,
        config: Config
    ):
        super().__init__(config)
