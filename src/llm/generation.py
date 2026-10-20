import torch

from llm.model.gpt import GPT


@torch.inference_mode()
def generate(
    model: GPT,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    temperature: float = 1.0,
    top_k: int | None = None,
    do_sample: bool = True,
    eos_token_id: int | None = None,
    use_cache: bool = True,
) -> torch.Tensor:
    model.eval()

    if max_new_tokens < 0:
        raise ValueError("max_new_tokens must be non-negative.")

    if do_sample and temperature <= 0:
        raise ValueError("temperature must be positive when sampling.")

    if top_k is not None and top_k <= 0:
        raise ValueError("top_k must be positive.")

    past_key_values = None

    for _ in range(max_new_tokens):
        if use_cache and past_key_values is not None:
            # Only process the newest token; previous K/V are cached.
            context = input_ids[:, -1:]
        else:
            # Without caching, recompute the available context.
            context = input_ids[:, -model.config.max_seq_len:]

        if use_cache:
            logits, past_key_values = model(
                context,
                past_key_values=past_key_values,
                use_cache=True,
            )
        else:
            logits = model(context)

        next_token_logits = logits[:, -1, :]

        if do_sample:
            next_token_logits = next_token_logits / temperature

            if top_k is not None:
                k = min(top_k, next_token_logits.size(-1))
                values, _ = torch.topk(next_token_logits, k)
                threshold = values[:, -1].unsqueeze(-1)
                next_token_logits = next_token_logits.masked_fill(
                    next_token_logits < threshold, float("-inf")
                )

            probs = torch.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
        else:
            next_token = torch.argmax(
                next_token_logits, dim=-1, keepdim=True
            )

        input_ids = torch.cat([input_ids, next_token], dim=1)

        if (
            eos_token_id is not None
            and torch.all(next_token.squeeze(-1) == eos_token_id)
        ):
            break

    return input_ids
