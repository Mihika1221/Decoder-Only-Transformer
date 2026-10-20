import torch 
import torch.nn as nn
class RotaryEmbedding(nn.Module):
    def __init__(
        self, 
        head_dim: int, 
        max_seq_len: int, 
        theta: float = 10000.0,
    ):
        super().__init__()
        if head_dim % 2 != 0:
            raise ValueError(
                f"head_dim({head_dim}) must e even for RoPE."
            
            )
        inv_freq = 1.0/(
            theta**(
                torch.arange(0, head_dim, 2, dtype= torch.float32)
                / head_dim
            )
        )
        positions = torch.arange(max_seq_len, dtype= torch.float32)
        freqs = torch.outer(positions, inv_freq)
        self.register_buffer("cos", freqs.cos(), persistent=False)
        self.register_buffer("sin", freqs.sin(), persistent=False)

    def forward(
        self,
        x: torch.Tensor,
        position_offset: int = 0,
    ) -> torch.Tensor:
        seq_len = x.size(-2)
        end_pos = position_offset + seq_len

        if end_pos > self.cos.size(0):
            raise ValueError("RoPE position exceeds maximum sequence length.")

        cos = self.cos[position_offset:end_pos].to(dtype=x.dtype)
        sin = self.sin[position_offset:end_pos].to(dtype=x.dtype)

        cos = cos.unsqueeze(0).unsqueeze(0)
        sin = sin.unsqueeze(0).unsqueeze(0)

        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]

        rotated_even = x_even * cos - x_odd * sin
        rotated_odd = x_even * sin + x_odd * cos

        return torch.stack(
            (rotated_even, rotated_odd), dim=-1
        ).flatten(-2)
