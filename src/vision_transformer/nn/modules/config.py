import logging


log = logging.getLogger(__name__)


class ViTConfig:

    def __init__(
        self,
        im_size=224,
        patch_size=16,
        qkv_bias=True,
        num_channels=3,
        block_size=1024,
        hidden_size=768,
        hidden_dropout_p=0.0,
        num_attention_heads=12,
        num_attention_layers=12,
        attention_dropout_prob=0.0,
    ) -> None:
        self.im_size = im_size
        self.patch_size = patch_size
        self.qkv_bias = qkv_bias
        self.num_channels = num_channels
        self.block_size = block_size
        self.hidden_size = hidden_size
        self.hidden_dropout_p = hidden_dropout_p
        self.num_attention_heads = num_attention_heads
        self.num_attention_layers = num_attention_layers
        self.attention_dropout_prob = attention_dropout_prob

        # Calculated Configs
        self.num_patches = (self.im_size // self.patch_size) ** 2
