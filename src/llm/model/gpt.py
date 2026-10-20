import torch
import torch.nn as nn
import torch.nn.functional as F

from llm.config import GPTConfig
from llm.model.block import TransformerBlock
from llm.model.norm import RMSNorm


class GPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config

        self.token_embedding = nn.Embedding(
            config.vocab_size, config.d_model
        )
        self.blocks = nn.ModuleList([
            TransformerBlock(config)
            for _ in range(config.n_layers)
        ])
        self.norm = RMSNorm(
            config.d_model, eps=config.rms_norm_eps
        )
        self.lm_head = nn.Linear(
            config.d_model, config.vocab_size, bias=False
        )

        self.apply(self._init_weights)
        self.lm_head.weight = self.token_embedding.weight

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: torch.Tensor | None = None,
        past_key_values=None,
        use_cache: bool = False,
    ):
        if input_ids.ndim != 2:
            raise ValueError(
                f"input_ids must have shape [B, T], got {tuple(input_ids.shape)}."
            )

        batch_size, seq_len = input_ids.shape
        past_len = (
            past_key_values[0][0].size(-2)
            if past_key_values is not None
            else 0
        )

        if past_len + seq_len > self.config.max_seq_len:
            raise ValueError(
                f"Total sequence length ({past_len + seq_len}) exceeds "
                f"max_seq_len ({self.config.max_seq_len})."
            )

        x = self.token_embedding(input_ids)
        presents = [] if use_cache else None

        if past_key_values is not None and len(past_key_values) != len(self.blocks):
            raise ValueError("One key/value cache is required per Transformer block.")

        for i, block in enumerate(self.blocks):
            layer_past = (
                past_key_values[i]
                if past_key_values is not None
                else None
            )

            if use_cache:
                x, present = block(
                    x,
                    past_key_value=layer_past,
                    use_cache=True,
                )
                presents.append(present)
            else:
                x = block(
                    x,
                    past_key_value=layer_past,
                    use_cache=False,
                )

        logits = self.lm_head(self.norm(x))

        if targets is not None:
            if past_key_values is not None:
                raise ValueError("Training loss is not supported with past_key_values.")
            if targets.shape != input_ids.shape:
                raise ValueError(
                    f"targets must match input_ids shape; got "
                    f"{tuple(targets.shape)} vs {tuple(input_ids.shape)}."
                )
            loss = F.cross_entropy(
                logits.reshape(-1, self.config.vocab_size),
                targets.reshape(-1),
            )
            if use_cache:
                return logits, loss, tuple(presents)
            return logits, loss

        if use_cache:
            return logits, tuple(presents)
        return logits
