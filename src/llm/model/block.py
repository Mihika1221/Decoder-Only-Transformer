import torch
import torch.nn as nn

from llm.config import GPTConfig
from llm.model.attention import CausalSelfAttention
from llm.model.mlp import SwiGLU
from llm.model.norm import RMSNorm


class TransformerBlock(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()

        self.attn_norm = RMSNorm(
            config.d_model,
            eps=config.rms_norm_eps,
        )
        self.attention = CausalSelfAttention(config)

        self.mlp_norm = RMSNorm(
            config.d_model,
            eps=config.rms_norm_eps,
        )
        self.mlp = SwiGLU(config)

    def forward(
        self,
        x: torch.Tensor,
        past_key_value: tuple[torch.Tensor, torch.Tensor] | None = None,
        use_cache: bool = False,
    ):
        if use_cache:
            attn_out, present = self.attention(
                self.attn_norm(x),
                past_key_value=past_key_value,
                use_cache=True,
            )
            x = x + attn_out
            x = x + self.mlp(self.mlp_norm(x))
            return x, present

        x = x + self.attention(self.attn_norm(x))
        x = x + self.mlp(self.mlp_norm(x))
        return x
